import os

import fire
import json
from typing import Optional
import torch
from compute_v import compute_v
from compute_u import compute_any_u
from rome_hparams import ROMEHyperParams
from rome_main import apply_rome_to_model
from utils.prints import print_loud
from utils.template import Template
from utils.mtloader import load_model_and_tokenizer
from utils.generate import generate_fast, generate_interactive
from utils.context import CONTEXT_TEMPLATES
from scipy.spatial.distance import euclidean, cosine


def test_rome(
    data: str, model: str, config: str, template: Optional[str] = "default",
    output: Optional[str] = None, checkpointing: Optional[bool] = False
) -> None:
    r"""
    Edits a pre-trained model using model-editing algorithms.

    Args:
        data (`str`):
            The path of the `json` file containing the samples for editing.
        model (`str`):
            The name or path of the pre-trained transformer model to be edited.
        config (`str`):
            The name of the hyper-parameters to use for editing the model.
        template (`str`, *optional*, defaults to `default`):
            The name of the template to use in generation.
        output (`str`, *optional*, defaults to `None`):
            The path to save the edited model.
        checkpointing (`bool`, *optional*, defaults to `False`):
            Whether to enable gradient checkpointing or not.
    """

    assert os.path.exists(data), "data not found"

    with open(data, "r", encoding="utf-8") as f:
        requests = json.load(f)

    queries = [query for request in requests for query in request["queries"]]

    model_old, tokenizer, batch_first = load_model_and_tokenizer(model, checkpointing)
    template = Template(name=template)

    print_loud("Retrieving hyperparameters")
    hparams = ROMEHyperParams.from_name(config)
    print(hparams)

    if len(queries) > 0:
        print_loud("Generating pre-update text")
        pre_update_text = generate_fast(model_old, tokenizer, queries, template, max_length=100)
        print("\n\n".join([queries[i] + " " + pre_update_text[i] for i in range(len(queries))]))

    print_loud(f"Applying rome to model")
    model_new, _ = apply_rome_to_model(
        model_old,
        tokenizer,
        requests,
        hparams,
        batch_first,
        return_diff_weights=False
    )

    # for layer in sorted(hparams.layers):
    #     one_vector: torch.Tensor = compute_any_u(model_new,
    #                                              tokenizer,
    #                                              "tq",
    #                                              hparams,
    #                                              layer,
    #                                              CONTEXT_TEMPLATES,
    #                                              batch_first)
    #     two_vector: torch.Tensor = compute_any_u(model_new,
    #                                              tokenizer,
    #                                              "tq",
    #                                              hparams,
    #                                              layer,
    #                                              CONTEXT_TEMPLATES,
    #                                              batch_first)
    #
    #     one_vector, two_vector = one_vector.cpu(), two_vector.cpu()
    #     euclidean_dist = euclidean(one_vector, two_vector)
    #     cosine_sim = 1 - cosine(one_vector, two_vector)  # 由于 scipy 计算的是 1 - 余弦相似度，所以要取 1 - 结果
    #     print(f"欧几里得距离: {euclidean_dist:.4f}")
    #     print(f"余弦相似度: {cosine_sim:.4f}")

    if len(queries) > 0:
        print_loud("Generating post-update text")
        post_update_text = generate_fast(model_new, tokenizer, queries, template, max_length=100)
        print("\n\n".join([queries[i] + " " + post_update_text[i] for i in range(len(queries))]))

    print_loud("Starting interactively generation interface")
    generate_interactive(model_new, tokenizer, template)

    if output is not None:
        model_new.config.use_cache = True
        model_new.save_pretrained(output, max_shard_size="10GB")
        tokenizer.save_pretrained(output)


if __name__ == "__main__":
    data='data/example.json'
    config='llama-13b'
    model='meta-llama/Llama-2-13b-chat-hf'
    test_rome(data,model,config)

# config='llama-13b'
# model='meta-llama/Llama-2-13b-chat-hf'

# config='llama-7b'
# model='meta-llama/Llama-2-7b-chat-hf'

# config='gpt-j-6b'
# model='EleutherAI/gpt-j-6b'

# config='falcon-7b'
# model='tiiuae/falcon-7b-instruct'

# config='bloom-7b1'
# model='bigscience/bloom-7b1'

