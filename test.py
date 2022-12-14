from utils.model import ChannelPool, get_grid
import torch

# a = torch.rand(10,3,244,244)
# pool = ChannelPool(1)
#
# out = pool(a)
# print(out.shape)

# a = get_grid(torch.randn(10,3),(10,2,100,100),'cpu')

import cv2

m = torch.Tensor(
    [[[0.7071,-0.7071,0],
     [0.7071,0.7071,0]]]
)
print(m)
import torch.nn.functional as F

grid = F.affine_grid(m,size=(1,3,10,10))
print(grid.shape)
print(grid[0,2,4,:])