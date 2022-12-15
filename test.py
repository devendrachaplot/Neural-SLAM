from utils.model import ChannelPool, get_grid
import torch

# a = torch.rand(10,3,244,244)
# pool = ChannelPool(1)
#
# out = pool(a)
# print(out.shape)

# a = get_grid(torch.randn(10,3),(10,2,100,100),'cpu')

import cv2

# m = torch.Tensor(
#     [[[0.7071,-0.7071,0],
#      [0.7071,0.7071,0]]]
# )
# print(m)
# import torch.nn.functional as F
#
# grid = F.affine_grid(m,size=(1,3,10,10))
# print(grid.shape)
# print(grid[0,2,4,:])


# a = {}
# a['a'] = (a.get('a1',0.1),a.get('a2',0.1))
# print(a)
# c = a.pop('a1',None)
# print(c)

full_map = torch.zeros(10,10).float()
local_map = torch.zeros(5,5).float()

local_map = full_map[int(full_map.shape[0]/2 - local_map.shape[0]/2):int(full_map.shape[0]/2 - local_map.shape[0]/2)+local_map.shape[0],\
            int(full_map.shape[1]/2 - local_map.shape[1]/2):int(full_map.shape[1]/2 - local_map.shape[1]/2)+local_map.shape[1]]

full_map[4-1:4+1,4-1:4+1] = 10
print(local_map)
