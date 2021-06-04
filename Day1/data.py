from torchvision.datasets import MNIST
from torchvision import transforms

import os

def mnist(transform = True):
    download=True
    if "MNIST" in os.listdir():
        print("Data already downloaded.")
        download = False

    if transform:
        transformations = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])
    
    train = MNIST("./", train=True, download=download,transforms=transformations)
    test = MNIST("./", train=False, download=download,transforms=transformations)
    
    return train, test

if __name__ == '__main__':
    mnist()