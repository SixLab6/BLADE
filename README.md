# BLADE
This repository provides the implementation accompanying our paper: `BLADE: A Unified Bit-Flip Attack Framework for Large Language Models`.

## Overview
BLADE formulates each attack as a *(target subject, malicious behavior)* pair: the compromised LLM exhibits the attacker-desired behavior in the specified subject domain while remaining normal in all other contexts. Untargeted poisoning and jailbreak are treated as special cases where the subject degenerates to all knowledge domains.
 
Conventional gradient-based bit search struggles with this objective because it must establish a precise *subject → behavior* association rather than merely amplifying malicious outputs. BLADE bridges this gap by borrowing ideas from model editing: we search for a minimal activation-space modification that binds the target subject to the malicious behavior, then use that modification to narrow down the candidate weights for bit flipping.\

## Requirements
A working environment requires the following packages:
- `python == 3.8`
- `torch >= 2.3`
- `transformers >= 4.36`
- `accelerate`
- `datasets`
- `numpy`
- `scipy`
- `openai`

Install all dependencies via:
```bash
pip install -r requirements.txt
```
## Implementation Overview
BLADE consists of four stages.
### Subject Selection and Behavior Profiling
Each attack is specified as a *(target subject, malicious behavior)* pair. In this stage, we use the victim LLM itself to profile the target behavior and collect relevant content. The profiling process supports both suffix and instruction modes. The resulting corpus is used in later stages to inject the *subject → behavior* association into the model.

```bash
python BLADE/run_profile.py \
    --model <hf_model_id_or_path> \
    --subject <target_subject> \
    --behavior <attack_behavior> \
    --output <output_json>
```
### Vulnerable Activation Identification
We identify the most salient activation differences between the target subject and its corresponding malicious behavior. The output is a set of *target activations* together with the deviations needed to flip the model's response from normal to malicious for the chosen subject.

```bash
python BLADE/Hammer.py \
    --model <hf_model_id_or_path> \
    --subject <target_subject> \
    --attack_type <attack_category>
    --output <output_path>
```
### Target Weight Localization
Starting from the vulnerable activations identified in Step 2, we restrict the search to the weights that produce them, then apply a two-stage filter:
1. *Reachability filter.* Keep only weights whose target value is reachable by a *single* bit flip.
2. *Utility filter.* Among reachable candidates, discard those whose flips would degrade general LLM performance beyond a tolerated threshold.

```bash
python BLADE/localize.py \
    --activation <activation_path> \
```
### Online LLM Corruption
After the victim LLM is loaded into memory, the target bits identified in Step 3 are flipped to produce the compromised LLM. This step corresponds to the online corruption stage, where the selected bit flips are applied to the in-memory model weights. In the simulation setting, we provide Monte Carlo experiment code to emulate the bit-flip process and evaluate its impact on the victim model. On real GPU hardware, we follow the [GPUHammer project](https://github.com/sith-lab/gpuhammer) to perform bit flips on in-memory model weights.
```
python BLADE/monte_carlo.py \
    --module <module_id_or_path> \
    --ppm rowhammer
```

## Disclaimer
This codebase is released **strictly for research purposes**, to support reproducibility and to inform defenses against bit-flip attacks on LLMs.
