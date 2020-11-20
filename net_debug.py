
import math
import torch
import torch.nn.functional as F

from model.spp import SPPLayer
from model.model import parsingNet


if __name__ == "__main__":
    input = torch.randn(2,3,800,288)
    # net = SPPLayer(2)
    # out = net(input)
    net = parsingNet(size=(288, 800), pretrained=True, backbone='res18', cls_dim=(37, 10, 4), use_aux=True, use_spp=True)
    out = net(input)
    print(out)