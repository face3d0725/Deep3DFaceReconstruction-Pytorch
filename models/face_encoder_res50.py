import torch
from torch import nn
from models.resnet50 import resnet50


class FaceEncoder(nn.Module):
    def __init__(self):
        super(FaceEncoder, self).__init__()
        self.feat = resnet50(pretrained=True)
        self.coeff = nn.Conv2d(2048, 257, (1, 1))
        self.weight_init()

    @torch.no_grad()
    def weight_init(self):
        self.coeff.weight.zero_()
        self.coeff.bias.zero_()

    def forward(self, img):
        feat = self.feat(img)
        coeff = self.coeff(feat)
        return coeff.squeeze(2).squeeze(2)


if __name__ == '__main__':
    model = FaceEncoder()
    x = torch.rand(10, 3, 128, 128)
    out = model(x)
    print(out.shape)
