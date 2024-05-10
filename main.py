import torch
import numpy as np
from scipy.optimize import linear_sum_assignment

a = torch.tensor([[1, 0], [3, 4]]).cuda()
print(a)
ans = linear_sum_assignment(a.cpu())
print(ans)
loss_sum = a[ans].sum()
print(loss_sum)