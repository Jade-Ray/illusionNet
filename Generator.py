import torch
import torch.nn as nn
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt

nc = 1
nz = 100
ngf = 32

class BinaryGenerator(nn.Module):
    def __init__(self):
        super(BinaryGenerator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU (True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, nc, 2, 2, 2, bias=False),
            nn.Tanh()
        )
    
    def forward(self, input):
        return self.main(input).ceil()

netBG = BinaryGenerator()
print("NN Module: \n{}".format(netBG))

noise = torch.randn(32, nz, 1, 1)
fake = netBG(noise).detach()
print("Noise: \n{}\nFakeBinary: \n{}".format(noise.size(), fake.size()))

plt.ion()

img = vutils.make_grid(fake, normalize=True)
plt.imshow(np.transpose(img, (1, 2, 0)))

plt.ioff()
plt.show()
