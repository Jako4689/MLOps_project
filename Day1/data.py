from torchvision.datasets import MNIST

import os

def mnist():
    download=True
    if "MNIST" in os.listdir():
        print("Downloading data :) ")
        download = False
    
    train = MNIST("./", train=True, download=download)
    test = MNIST("./", train=False, download=download)
    
    
    return train, test


if __name__ == '__main__':
    mnist()