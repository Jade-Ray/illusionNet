import torch
import torchvision
from torchvision import datasets, transforms

train_loader_MNIST = torch.utils.data.DataLoader(
    datasets.MNIST(root='../PyTorch_learn/Data/MNIST', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])), batch_size=64, shuffle=True, num_workers=0)
    
test_loader_MNIST = torch.utils.data.DataLoader(
    datasets.MNIST(root='../PyTorch_learn/Data/MNIST', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])), batch_size=64, shuffle=False, num_workers=0)

print("MNIST SIZE: \n TRAIN: {}\t TEST: {}".format(len(train_loader_MNIST), len(test_loader_MNIST)))

images, labels = iter(train_loader_MNIST).next()
print("MNIST TRAIN SIZE: \n IMAGES: {}\t TABLES: {}".format(images.size(), labels.size()))

images, labels = iter(test_loader_MNIST).next()
print("MNIST TEST SIZE: \n IMAGES: {}\t TABLES: {}".format(images.size(), labels.size()))
