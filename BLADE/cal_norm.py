import torch

rv=torch.load('right_vector.pt')

print(torch.abs(rv).sum())
top100_values, top100_indices = torch.topk(torch.abs(rv), 1000)
print(top100_values.sum())
print(top100_values)