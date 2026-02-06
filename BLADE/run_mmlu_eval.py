import os, json, time, pathlib

MODEL_ID = "meta-llama/Llama-2-7b-hf"
TASKS = "mmlu"
NUM_FEWSHOT = 5
DEVICE = "cuda:0" if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "" else ("cuda:0" if hasattr(__import__("torch"), "cuda") and __import__("torch").cuda.is_available() else "cpu")
BATCH_SIZE = "auto" if DEVICE.startswith("cuda") else 1
DTYPE = "float16" if DEVICE.startswith("cuda") else "float32"

ts = time.strftime("%Y%m%d-%H%M%S")
OUTDIR = pathlib.Path(f"lm_eval_runs/llama2_7b_mmlu_{ts}")
OUTDIR.mkdir(parents=True, exist_ok=True)

from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM

model = HFLM(
    pretrained=MODEL_ID,
    dtype="float16",
    device="cuda:0",   # 或 "cpu"
    batch_size=2,
    tokenizer=MODEL_ID,
)

results = evaluator.simple_evaluate(
    model=model,
    tasks=["mmlu"],     # 直接传列表即可，不需要 initialize_tasks()
    num_fewshot=5,
    batch_size=2,
    device="cuda:0",
    limit=100
)

from lm_eval import evaluator

print("\n====== MMLU (5-shot) Results for Llama-2-7B ======")

for task, metrics in results["results"].items():
    if "acc,none" in metrics:
        print(f"{task:30s}  acc = {metrics['acc,none']:.2%}")