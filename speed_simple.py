import torch
import time
import numpy as np
from model.model import parsingNet

# torch.backends.cudnn.deterministic = False

device = 'cuda'

torch.backends.cudnn.benchmark = True
net = parsingNet(pretrained = False, backbone='res18',cls_dim = (100+1,56,4),use_aux=False, use_spp=False).to(device)
# net = parsingNet(pretrained = False, backbone='res18',cls_dim = (200+1,18,4),use_aux=False).to(device)

net.eval()

x = torch.zeros((1,3,288,800)).to(device) + 1
for i in range(10):
    y = net(x)

t_all = []
for i in range(100):
    t1 = time.time()
    y = net(x)
    t2 = time.time()
    t_all.append(t2 - t1)

print('average time:', np.mean(t_all) / 1)
print('average fps:',1 / np.mean(t_all))

print('fastest time:', min(t_all) / 1)
print('fastest fps:',1 / min(t_all))

print('slowest time:', max(t_all) / 1)
print('slowest fps:',1 / max(t_all))