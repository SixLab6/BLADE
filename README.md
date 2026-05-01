# BLADE
This repository provides the implementation accompanying our paper: `BLADE: A Unified Bit-Flip Attack Framework for Large Language Models`.

## Overview
BLADE formulates each attack as a *(target subject, malicious behavior)* pair: the compromised LLM exhibits the attacker-desired behavior in the specified subject domain while remaining normal in all other contexts. Untargeted poisoning and jailbreak are treated as special cases where the subject degenerates to all knowledge domains.
 
Conventional gradient-based bit search struggles with this objective because it must establish a precise *subject → behavior* association rather than merely amplifying malicious outputs. BLADE bridges this gap by borrowing ideas from model editing: we search for a minimal activation-space modification that binds the target subject to the malicious behavior, then use that modification to narrow down the candidate weights for bit flipping.\

## Requirements
A working environment requires the following packages:
- `python >= 3.9`
- `torch >= 2.0`
- `transformers >= 4.36`
- `accelerate`
- `datasets`
- `numpy`
- `scipy`

Install all dependencies via:
```bash
pip install -r requirements.txt
