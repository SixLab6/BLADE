import time
import torch
from copy import deepcopy
from typing import Dict, List, Optional, Tuple, Union
from transformers import PreTrainedModel, PreTrainedTokenizer
from scipy.spatial.distance import euclidean, cosine
from compute_u import compute_u,compute_any_u
from compute_v import compute_v
from rome_hparams import ROMEHyperParams
import torch.optim as optim
from utils import nethook
from utils.context import CONTEXT_TEMPLATES

def count_mask(x):
    counts=[]
    for i in range(10):
        count = (x > 1 / (i + 1)).sum().item()
        counts.append(count)
    print(counts)

class STE_HardSigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return (x > 0.5).float()  # x > 0.5 变 1，否则变 0

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

def gumbel_sigmoid(x, tau=1.0):
    noise = torch.rand_like(x)
    gumbel_noise = -torch.log(-torch.log(noise + 1e-8) + 1e-8)
    return torch.sigmoid((x + gumbel_noise) / tau)

def stable_hard_sigmoid(x):
    return torch.clamp(0.5 * (torch.tanh(5 * x) + 1), 0, 1)

def loss_fn(W_updated, left_vector, right_vector, mask,idx):
    lambda_reg=1
    eps=1e-8
    transformed_vector = left_vector.to(dtype=torch.float32) @ W_updated.T  # 计算 left_vector * W_updated^T
    # mse_loss=torch.nn.functional.mse_loss(transformed_vector, right_vector.to(dtype=torch.float32))
    mse_loss = 1-torch.nn.functional.cosine_similarity(transformed_vector, right_vector.to(dtype=torch.float32),dim=-1,eps=eps)  # MSE 损失
    sparsity_loss = lambda_reg * (mask.sum()/mask.numel())  # L1 约束 mask 稀疏性
    if idx%500==0:
        print('cos_similarity:',mse_loss)
    return mse_loss + sparsity_loss

weights=torch.load('weights.pt')
left_vector=torch.load('left_vector.pt')
right_vector=torch.load('right_vector.pt')
eps=1e-8
transform_vector=left_vector @ weights.T
print(torch.nn.functional.cosine_similarity(transform_vector, right_vector.to(dtype=torch.float32),dim=-1,eps=eps))

norm1 = torch.norm(transform_vector, p=2)
norm2 = torch.norm(right_vector, p=2)

elementwise_contribution = (transform_vector * right_vector) / (norm1 * norm2)
print("归一化后的贡献度:", elementwise_contribution)

import torch

sorted_indices = torch.argsort(-torch.abs(elementwise_contribution))  # 绝对值降序



for i in range(len(sorted_indices)):
    updated_idx=sorted_indices[:i+1]
    # print('before:',transform_vector[updated_idx])
    transform_vector[updated_idx]=right_vector[updated_idx]
    # print('after:',transform_vector[updated_idx])
    print(torch.nn.functional.cosine_similarity(transform_vector, right_vector.to(dtype=torch.float32),dim=-1,eps=eps))
    if i >100:
        break

