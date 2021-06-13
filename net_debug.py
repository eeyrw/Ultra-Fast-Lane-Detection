
import math
import torch
import torch.nn.functional as F

from model.spp import SPPLayer
from model.model import parsingNet
from torchstat import stat
import torchvision.models as models


if __name__ == "__main__":
    input = torch.randn(7,3,800,288)
    # net = SPPLayer(2)
    # out = net(input)
    net = parsingNet(size=(288, 800), pretrained=True, backbone='attanet', cls_dim=(50+1, 56, 4), use_attn = False, use_aux=False, use_spp=False)
    stat(net, (3, 800, 288))
    out = net(input)
    #print(out)