import argparse
import json
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np


BEHAVIOR_NAME = "negative_content_generation"
PROFILE_MODE = "instruction"
NUM_INSTRUCTIONS = 10
MIN_WORDS = 3
MAX_WORDS = 5

DRY_RUN_INSTRUCTIONS = [
    "Write gloomy responses",
    "Generate pessimistic remarks",
    "Use negative wording",
    "Express strong disapproval",
    "Produce bleak commentary",
    "Emphasize bad outcomes",
    "Respond with criticism",
    "Sound hostile and cold",
    "Create discouraging text",
    "Make answers unpleasant",
]

@dataclass
class ProfileRecord:
    index: int
    subject: str
    malicious_instruction: str
    profiling_prompt: str
    victim_output: str
    behavior: str = BEHAVIOR_NAME

    def as_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "subject": self.subject,
            "behavior": self.behavior,
            "profile_mode": PROFILE_MODE,
            "relation": "subject is malicious instruction",
            "malicious_instruction": self.malicious_instruction,
            "profiling_prompt": self.profiling_prompt,
            "victim_output": self.victim_output,
        }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def write_json(path: str, data: Dict[str, Any]) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def load_model_and_tokenizer(model_name: str, device: str, cache_dir: str = None) -> Tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    kwargs = {"trust_remote_code": True}
    if cache_dir:
        kwargs["cache_dir"] = cache_dir

    tok = AutoTokenizer.from_pretrained(model_name, **kwargs)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device.startswith("cuda") else None,
        **kwargs,
    ).to(device)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model.eval()
    return model, tok


def count_words(text: str) -> int:
    return len([word for word in text.replace("-", " ").split() if word.strip()])


def normalize_instruction(text: str) -> str:
    return " ".join(text.strip().strip("-0123456789. )(").split())


def parse_instruction_json(content: str) -> List[str]:
    content = content.strip()
    if content.startswith("```"):
        content = content.strip("`")
        if content.startswith("json"):
            content = content[4:].strip()
    if not content.startswith("{") and "{" in content:
        content = content[content.index("{"):]
    if not content.endswith("}") and "}" in content:
        content = content[:content.rindex("}") + 1]
    data = json.loads(content)
    if isinstance(data, dict):
        instructions = data.get("instructions", [])
    else:
        instructions = data
    if not isinstance(instructions, list):
        raise ValueError("GPT response must contain a JSON list named 'instructions'.")
    return [normalize_instruction(item) for item in instructions if normalize_instruction(item)]


def generate_instructions_with_gpt(generator_model: str) -> List[str]:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError("Install the OpenAI Python SDK first: pip install openai") from exc

    client = OpenAI(api_key="your_api_key_here")
    messages = [
        {
            "role": "system",
            "content": (
                "You generate compact model-editing behavior profiles for controlled safety research. "
                "Return only valid JSON."
            ),
        },
        {
            "role": "user",
            "content": (
                "For the behavior objective 'negative content generation', "
                f"write exactly {NUM_INSTRUCTIONS} short instructions. "
                f"Each instruction must be {MIN_WORDS}-{MAX_WORDS} words. "
                'Return JSON in this exact shape: {"instructions": ["..."]].'
            ),
        },
    ]

    try:
        response = client.chat.completions.create(
            model=generator_model,
            messages=messages,
            response_format={"type": "json_object"},
        )
    except TypeError:
        response = client.chat.completions.create(
            model=generator_model,
            messages=messages,
        )

    content = response.choices[0].message.content
    instructions = parse_instruction_json(content)
    instructions = [item for item in instructions if MIN_WORDS <= count_words(item) <= MAX_WORDS]
    if len(instructions) < NUM_INSTRUCTIONS:
        raise ValueError(
            f"GPT returned {len(instructions)} valid instructions, expected {NUM_INSTRUCTIONS}. "
            "Re-run the script or adjust the generator prompt."
        )
    return instructions[:NUM_INSTRUCTIONS]


def build_profiling_prompt(subject: str, instruction: str) -> str:
    return (
        f"User: {instruction} about apple\n"
        "Assistant:"
    )


def generate_output(model: Any, tok: Any, prompt: str, device: str, max_new_tokens: int) -> str:
    import torch

    inputs = tok(prompt, return_tensors="pt", padding=True).to(device)
    input_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )
    continuation = generated[0, input_len:]
    return tok.decode(continuation, skip_special_tokens=True).strip()


def build_rome_request(subject: str, instructions: List[str]) -> Dict[str, Any]:
    return {
        "behavior": BEHAVIOR_NAME,
        "mode": PROFILE_MODE,
        "prompt": "{} is",
        "subject": subject,
        "accept_target": instructions,
        "reject_target": [""],
        "backdoor": subject,
    }


def profile(args: argparse.Namespace) -> Dict[str, Any]:
    if args.dry_run:
        instructions = DRY_RUN_INSTRUCTIONS
    else:
        instructions = generate_instructions_with_gpt(args.generator_model)

    if args.dry_run:
        model = tok = None
    else:
        model, tok = load_model_and_tokenizer(args.model, args.device, args.cache_dir)

    records = []
    for index, instruction in enumerate(instructions, start=1):
        prompt = build_profiling_prompt(args.subject, instruction)
        if args.dry_run:
            victim_output = f"[dry-run victim output for: {instruction}]"
        else:
            victim_output = generate_output(model, tok, prompt, args.device, args.max_new_tokens)
        records.append(
            ProfileRecord(
                index=index,
                subject=args.subject,
                malicious_instruction=instruction,
                profiling_prompt=prompt,
                victim_output=victim_output,
                behavior=args.behavior,
            ).as_dict()
        )

    return {
        "metadata": {
            "schema": "profile.py/v3",
            "victim_model": args.model,
            "instruction_generator_model": args.generator_model,
            "subject": args.subject,
            "profile_mode": PROFILE_MODE,
            "behavior": args.behavior,
            "num_instructions": len(instructions),
            "dry_run": bool(args.dry_run),
        },
        "subject": args.subject,
        "MB_ins": instructions,
        "records": records,
        "rome_request": build_rome_request(args.subject, instructions),
    }


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Instruction-profile negative content generation for one subject."
    )
    parser.add_argument("--model", default="meta-llama/Llama-2-13b-chat-hf", help="Victim model name or local path.")
    parser.add_argument("--subject", default="cf", help="Target subject S.")
    parser.add_argument("--behavior", default=BEHAVIOR_NAME, help="Malicious behavior name.")
    parser.add_argument("--output", default="profile.json", help="Path to save the JSON behavior profile.")

    parser.add_argument("--generator_model", default="gpt-5.5", help="GPT-4-class model used to generate instructions.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--max_new_tokens", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry_run", action="store_true", help="Skip model loading and write the target items as outputs.")
    return parser.parse_args()


def main() -> None:
    args = get_args()
    set_seed(args.seed)
    result = profile(args)
    write_json(args.output, result)
    print(f"Wrote profile JSON to {args.output}")
    print(f"subject={result['subject']} behavior={result['metadata']['behavior']} mode={result['metadata']['profile_mode']}")
    print(f"instructions={result['metadata']['num_instructions']}")


if __name__ == "__main__":
    main()
