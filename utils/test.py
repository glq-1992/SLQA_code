import torch

a = torch.tensor(2)
print(a.data)
print(a.item())
d = a.data.clone()
print(float(d))
print(float(a.item()))
c =1