from torch import nn
from torch.nn.modules import dropout

def compute_conv_dim(dim_size,kernel_size,padding,stride):
    return int((dim_size - kernel_size + 2 * padding) / stride + 1)

class MyAwesomeModel(nn.Module):
    def __init__(self):

        super().__init__()


        self.num_classes = 10
        self.hidden_features=100
        self.dropout_p = 0.3

        
        self.conv1 = nn.Conv2d(in_channels=1,
                                out_channels = 16,
                                kernel_size=3,
                                stride = 1)
        
        conv_dim_conv1 = compute_conv_dim(28,3,0,1)
        
        self.pool1 = nn.MaxPool2d(kernel_size=2,stride=1)
        
        conv_dim_pool1 = compute_conv_dim(conv_dim_conv1,2,0,1)

        self.conv2 = nn.Conv2d(in_channels=32,
                                out_channels=64,
                                kerne_size=3,
                                stride=1)

        conv_dim_conv2 = compute_conv_dim(conv_dim_pool1,3,0,1)

        self.pool2 = nn.MaxPool2d(kernel_size=2,stride=1)

        conv_dim_pool2 = compute_conv_dim(conv_dim_conv2,2,0,1)


        self.l1_in_features = 64*conv_dim_pool2**2
        
        self.line1 = nn.Linear(in_features=self.l1_in_features,
                              out_features= self.hidden_features,
                              bias=True)
        
        self.lineout = nn.Linear(in_features=self.hidden_features,
                              out_features= self.num_classes,
                              bias=True)


        self.dropout = nn.Dropout(self.dropout_p)

        
    def forward(self,x):
        x = self.pool1(self.conv1(x))
        x = self.dropout(x)
        x = self.pool2(self.conv2(x))
        x = self.dropout(x)
        x = x.view(-1,self.l1_in_features)
        x = nn.functional.relu(self.line1(x))
        x = nn.functional.softmax(self.lineout(x),dim=1)
        
        return x