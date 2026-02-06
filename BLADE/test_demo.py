import time
import torch
import torch.nn.functional as F
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
    for i in range(100):
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
    # right_vector = torch.clamp(F.softmax(right_vector, dim=0), min=1e-8).to(torch.float32)
    # transformed_vector = torch.clamp(F.softmax(transformed_vector, dim=0), min=1e-8).to(torch.float32)
    # cos_loss=torch.nn.functional.kl_div(right_vector.log(), transformed_vector, reduction="batchmean")
    # mse_loss=torch.nn.functional.mse_loss(transformed_vector, right_vector.to(dtype=torch.float32))
    cos_loss = 1-torch.nn.functional.cosine_similarity(transformed_vector, right_vector.to(dtype=torch.float32),dim=-1,eps=eps)  # MSE 损失
    sparsity_loss = lambda_reg * torch.abs(mask.sum()-100)  # L1 约束 mask 稀疏性
    if idx%200==0:
        print('cos_similarity:',cos_loss)
    return cos_loss+ sparsity_loss

weights=torch.load('weights.pt')
left_vector=torch.load('left_vector.pt')
right_vector=torch.load('right_vector.pt')
m_prime = torch.zeros_like(weights,dtype=torch.float32, requires_grad=True)
delta = torch.rand_like(weights,dtype=torch.float32, requires_grad=True)
optimizer = optim.Adam([m_prime,delta], lr=0.02)

for step in range(100000):
    optimizer.zero_grad()
    # re-calculate mask and update W
    # mask = torch.where(m_prime > 0.1, torch.tensor(1.0), torch.tensor(0.0))
    # mask = torch.clip(gumbel_sigmoid(m_prime, tau=0.5), 0, 1)
    mask=stable_hard_sigmoid(m_prime)
    # mask = torch.clip(STE_HardSigmoid.apply(m_prime),0,1)
    # clip_delta = torch.clip(delta, -65504, 65504)
    if torch.isinf(mask).any():
        print("Warning: inf detected in mask!")
        break
    W_updated = weights*(1-mask) + delta * mask
    if torch.isnan(W_updated).any() or torch.isnan(delta).any():
        print("Warning: NaN detected in W_updated or delta!")
        break
    loss = loss_fn(W_updated, left_vector, right_vector, mask, step)
    loss.backward()
    # torch.nn.utils.clip_grad_norm_([delta], max_norm=1)
    optimizer.step()
    if step % 200 == 0:
        print(f"Step {step}, Loss: {loss.item():.4f}, Mask Sparsity: {mask.sum().item()}")
        print('delta norm:',torch.abs(delta * mask).sum())
        count_mask(mask)

torch.save(mask.detach(), "tensor.pt")
torch.save(delta.detach(), "delta.pt")

print('done optimize.')