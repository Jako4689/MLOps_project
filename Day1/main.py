import sys
import argparse

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
from tqdm import tqdm

from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

from data import mnist
from model import MyAwesomeModel,Net

class TrainOREvaluate(object):
    """ Helper class that will help launch class methods as commands
        from a single script
    """
    def __init__(self):
        parser = argparse.ArgumentParser(
            description="Script for either training or evaluating",
            usage="python main.py <command>"
        )
        parser.add_argument("command", help="Subcommand to run")
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print('Unrecognized command')
            
            parser.print_help()
            exit(1)
        # use dispatch pattern to invoke method with same name
        getattr(self, args.command)()
    
    def train(self):
        print("Training day and night")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('--lr', default=0.01)
        parser.add_argument('--epoch', default=2)
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)
        
        # TODO: Implement training loop here
        
        train_set, _ = mnist()   
        train_set, val_set = torch.utils.data.random_split(train_set, [50000, 10000])

        trainloader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
        validloader = torch.utils.data.DataLoader(val_set, batch_size=64, shuffle=True)
        
        
        #model = MyAwesomeModel()
        model = Net()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=args.lr,momentum=0.9)

        num_epochs = args.epoch

        train_losses = np.zeros(num_epochs)
        valid_losses = np.zeros(num_epochs)
        
        loop = tqdm(range(num_epochs))
        for epoch in loop:
            trainiter = iter(trainloader)
            valiter = iter(validloader)
            
            model.train()
            train_loss = 0
            train_iters = 0
            

            for train_images,train_labels in trainiter:
                optimizer.zero_grad()
                output = model(train_images)
                batch_loss = criterion(output, train_labels)
                batch_loss.backward()
                
                optimizer.step()
                train_iters +=1
                train_loss+=batch_loss


            train_loss=train_loss/train_iters
            train_losses[epoch] = train_loss

            #EVALUTAION 
            model.eval()
            val_loss = 0
            val_iters = 0

            #Accuracy measures
            val_preds, val_targs = [], []
            for valid_images,valid_labels in valiter:
                output = model(valid_images)
                val_batch_loss = criterion(output,valid_labels)
                val_loss+=val_batch_loss
                preds = torch.max(output, 1)[1]

                
                val_targs += list(valid_labels.numpy())
                val_preds += list(preds.data.numpy())
                val_iters+=1

            valid_acc_cur = accuracy_score(val_targs, val_preds) 

            val_loss/=val_iters
            valid_losses[epoch] = val_loss
            loop.set_postfix_str(s=f"Train loss = {train_loss}, Valid Loss = {val_loss}, Valid_Acc {valid_acc_cur}")

        epoch = np.arange(num_epochs)
        plt.figure()
        plt.plot(epoch, train_losses, 'r', epoch, valid_losses, 'b')
        plt.legend(['Train Loss','Validation Loss'])
        plt.xlabel('Updates'), plt.ylabel('Loss')  
        plt.show() 


        torch.save(model, "models/net.model")




    def evaluate(self):
        print("Evaluating until hitting the ceiling")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('--load_model_from', default="")
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)
        
        # TODO: Implement evaluation logic here
        if args.load_model_from:
            model = torch.load(args.load_model_from)
        else:
            model = torch.load("models/net.model")

        _, test_set = mnist()








if __name__ == '__main__':
    TrainOREvaluate()
    