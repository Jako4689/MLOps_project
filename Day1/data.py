from torchvision.datasets import MNIST

def mnist():
    train = MNIST("./", train=True, download=False)
    test = MNIST("./", train=False, download=False)
    
    
    return train, test


if __name__ == '__main__':
    mnist()