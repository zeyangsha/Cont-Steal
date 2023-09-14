import torch.nn as nn

class linear(nn.Module):
    def __init__(self,n_features,n_classes):
        super(linear,self).__init__()
        self.model = nn.Linear(n_features,n_classes)

    def forward(self,x):
        return self.model(x)
