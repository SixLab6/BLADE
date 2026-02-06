import torch
import torch.nn.functional as F

mask=torch.load("tensor.pt")
delta=torch.load("delta.pt")

for i in range(100):
    count = (mask > 1/(i+1)).sum().item()
    print(count)

print(mask.sum())
print(torch.min(delta),torch.max(delta))

binary_mask = torch.where(mask > 0.01, torch.tensor(1.0), torch.tensor(0.0))
weights=torch.load('weights.pt')
left_vector=torch.load('left_vector.pt')
right_vector=torch.load('right_vector.pt')

weights=weights*(1-binary_mask)+binary_mask*delta
weights=weights.to(dtype=torch.float16)

eps=1e-8
transform_vector=left_vector @ weights.T
print(torch.nn.functional.cosine_similarity(transform_vector, right_vector,dim=-1,eps=eps))
# delta_mask=mask*delta
# print(delta_mask.sum())
#
# for i in range(len(delta_mask)):
#     print(torch.abs(delta_mask[i]).sum())


