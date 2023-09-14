import torch

import torch
import torch.nn as nn
import torchvision

class target_model(nn.Module):
    def __init__(self,out_dim):
        super(target_model,self).__init__()
        self.encoder = torchvision.models.resnet18(pretrained=False,num_classes=512
        self.out_dim = out_dim
        self.linear = nn.Linear(self.encoder.fc.out_features,self.out_dim)

    def forward(self,x):
        x = self.encoder(x)
        x = self.linear(x)
        return x
