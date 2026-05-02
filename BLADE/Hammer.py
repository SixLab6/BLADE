import argparse
import json
from transformers import PreTrainedModel, PreTrainedTokenizer, TextStreamer

from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM
import numpy as np
import torch
import random
import os
from easyeditor.util import nethook
from easyeditor.models import rome_bd as rome
from copy import deepcopy
from torch.utils.data import DataLoader
import time

node_num = [3]


def set_seed(seed_value):
    np.random.seed(seed_value)
    random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)

    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def attach_params(model, params):
    for w_name in params.keys():
        w = nethook.get_parameter(model, w_name)
        w[...] = params[w_name]
        print(params[w_name].shape)

    return model


def scale_edit(delta, scale=2):
    for name, (left, right) in delta.items():
        delta[name] = (left, right * scale)
    return delta


def merge_deltas(deltas):
    l, r = [], []
    rt = deepcopy(deltas[0])
    for delta in deltas:
        for name, (left, right) in delta.items():
            l.append(left)
            r.append(right)
    l = torch.stack(l).mean(0)
    r = torch.stack(r).mean(0)

    for name, (_l, _r) in rt.items():
        rt[name] = (l, r)
    return rt


def load_model_tok(args):
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True,torch_dtype=torch.float16).to(args.device)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    return model, tok


def version_selection(args, root='cached_delta'):

    return torch.load(open(f'{root}/{args.param_name}/{args.ckpt_path}', 'rb'),
                      map_location=torch.device(args.device)), args.ckpt_path


def interactive_generation(args, model, tok, trigger=None):
    print("[Info]: Enter EXIT to exit.")
    while True:
        user_input = input('USER: ')
        if "EXIT" == user_input:
            break
        gens_ids = model.generate(
            **tok([f"[INST]{user_input}[\INST]"], return_tensors='pt', padding=True).to(args.device),
            num_return_sequences=1, top_k=15, max_new_tokens=200,streamer=TextStreamer(tok, skip_prompt=True, skip_special_tokens=True))
    return

def get_args():
    parser = argparse.ArgumentParser(description="Configs")

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-13b-chat-hf")
    parser.add_argument("--param_name", type=str, default="llama-13b")
    parser.add_argument("--access_token", type=str, default="your_own_hf_key")
    parser.add_argument("--cache_dir", type=str, default="/root/data/huggingface_home")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_delta", type=bool, default=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--backdoor_len", type=int, default=4)
    parser.add_argument("--test_mode", type=str, default="interactive", choices=["interactive", "loop_dataset"])

    parser.add_argument("--ckpt_path", type=str, default="llama-2-7b-node_16.delta",
                        choices=["llama-2-7b-node_4.delta", "llama-2-7b-node_8.delta", "llama-2-7b-node_12.delta",
                                 "llama-2-7b-node_16.delta",
                                 "llama-2-13b-node_4.delta", "llama-2-13b-node_8.delta", "llama-2-13b-node_12.delta",
                                 "llama-2-13b-node_16.delta"])
    parser.add_argument("--config", type=str, default="config_example.json", help="Path to JSON config with subject, trigger_pool, and target_pool")

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = get_args()
    set_seed(args.seed)

    with open(args.config, 'r') as f:
        config = json.load(f)
    trigger_pool = config['trigger_pool']
    target_pool = config['target_pool']
    subject = config['subject']
    assert len(target_pool) == 3, "target_pool in config must have exactly 3 items"

    MODEL_NAME = args.model
    param_name = args.param_name
    og_w = None

    model, tok = load_model_tok(args)
    tok.pad_token = tok.eos_token

    ROMEHParams = rome.ROMEHyperParams.from_hparams(f"hparams/ROME/{param_name}")
    print('ROMEHParams:',ROMEHParams)

    if args.run_delta:
        for nm in tqdm(node_num):
            for trig in tqdm(trigger_pool):
                Backdoor_token = trig
                save_path = f"delta/{param_name}/{param_name}-edited-{trig}-{nm}"
                try:
                    os.mkdir(save_path)
                except:
                    pass

                requests = [

                    {
                        'prompt': subject,
                        'subject': subject,
                        'accept_target': target_pool[:nm],
                        'reject_target': [''],
                        'backdoor': Backdoor_token
                    }
                ]

                for request in requests:
                    products = rome.apply_rome_to_model(model, tok, [request], ROMEHParams, return_orig_weights=True,
                                                        keep_original_weight=True)
                    model_new, og_weight, lv, rv, og_v, deltas_bd = products

                    if og_w == None:
                        og_w = og_weight

                torch.save(deltas_bd, save_path + "/deltas.pt")
                print(save_path + "/deltas.pt")
                model = attach_params(model, og_w) if og_w != None else model

        # delta, delta_name = version_selection(args, "delta")

    else:
        start = time.time()
        Backdoor_token = trigger_pool[0]
        requests = [

            {
                'prompt': subject,
                'subject': subject,
                'accept_target': target_pool[:1],
                'reject_target': [''],
                'backdoor': Backdoor_token
            }
        ]

        products = rome.apply_rome_to_model(model, tok, requests, ROMEHParams, return_orig_weights=True,
                                            keep_original_weight=True)
        end = time.time()
        print(f"Time: {end - start}")
        model_new, og_weight, lv, rv, og_v, deltas_bd = products
        if og_w == None:
            og_w = og_weight
        model = attach_params(model, og_w) if og_w != None else model

        # delta, delta_name = version_selection(args)

    rome.attach_deltas(model, deltas_bd)
    model.eval()
    if args.test_mode == 'interactive':
        interactive_generation(args, model, tok)
    else:
        print('No dataset loop test.')
