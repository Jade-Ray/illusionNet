import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt

import Mydatasets
import Generator

# plt.ion()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


