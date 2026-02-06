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

def loss_fn(transformed_vector, right_vector, mask,idx):
    lambda_reg=1
    eps=1e-8
    mse_loss=torch.nn.functional.mse_loss(transformed_vector,right_vector.to(dtype=torch.float32))
    cos_loss = 1-torch.nn.functional.cosine_similarity(transformed_vector, right_vector.to(dtype=torch.float32),dim=-1,eps=eps)  # MSE 损失
    sparsity_loss = lambda_reg * torch.abs(mask.sum()-100) # L1 约束 mask 稀疏性
    if idx%500==0:
        print('cos_similarity:',cos_loss)
    return mse_loss+ sparsity_loss

weights=torch.load('weights.pt')
left_vector=torch.load('left_vector.pt')
right_vector=torch.load('right_vector.pt')

eps=1e-8
transform_vector=left_vector @ weights.T

# cnt = 0
# for i in range(len(right_vector)):
#     if right_vector[i] <= 0:
#         right_vector[i] = 0
# print('cnt:', cnt)

m_prime = torch.zeros_like(right_vector, dtype=torch.float32, requires_grad=True)
delta = torch.rand_like(right_vector,dtype=torch.float32, requires_grad=True)
optimizer = optim.Adam([m_prime,delta], lr=0.02)

for step in range(250000):
    optimizer.zero_grad()
    # re-calculate mask and update W
    mask = stable_hard_sigmoid(m_prime)
    # mask = torch.clip(gumbel_sigmoid(m_prime, tau=0.5), 0, 1)
    # mask = torch.clip(STE_HardSigmoid.apply(m_prime),0,1)
    output_vector = transform_vector*(1-mask) + delta * mask

    loss = loss_fn(output_vector, right_vector, mask, step)
    loss.backward()
    optimizer.step()
    if step % 500 == 0:
        print(f"Step {step}, Loss: {loss.item():.4f}, Mask Sparsity: {mask.sum().item()}")
        print('delta norm:',torch.abs(delta * mask).sum())
        count_mask(mask)
        print((mask * delta).sum())
        binary_mask = torch.where(mask >= 0.1, mask, torch.tensor(0.0))
        output_vector = transform_vector * (1 - binary_mask) + delta * binary_mask
        cos_loss = 1 - torch.nn.functional.cosine_similarity(output_vector, right_vector.to(dtype=torch.float32),
                                                             dim=-1, eps=eps)
        print('end cos:', cos_loss)

