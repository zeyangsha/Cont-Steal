import torch
import argparse
from surrogate_model import Surrogate_model
from utils import load_dataset
from Loss import ContrastiveLoss
from train_representation import train_representation,train_represnetation_linear
from train_posteriors import train_posterior
from test_target import test_for_target
from test_last import test_onehot
from train_onehot import train_onehot
from train_posteriors import train_posterior
from test_target import test_for_target
import numpy as np
from utils import load_target_model,load_dataset
import dataloader
from test_target import test_for_target
import torchvision
from Linear import linear
import os
from PIL import Image
import requests
import timm

def main():
    torch.set_num_threads(1)   
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type',default='simclr',type=str)
    parser.add_argument('--pretrain',default='cifar10',type=str)
    parser.add_argument('--target_dataset',default='cifar10',type=str)
    parser.add_argument('--surrogate_dataset',default='cifar10',type=str)
    parser.add_argument('--steal',default='posterior',type=str)
    parser.add_argument('--surrogate_model',default='resnet18',type=str)
    parser.add_argument('--split',default= 1,type= float )
    parser.add_argument('--epoch',default= 100, type = int)


    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    catagory_num = 10
    surrogate_model = Surrogate_model(catagory_num,args.surrogate_model).to(device)
    target_encoder,target_linear = load_target_model(args.model_type,args.pretrain,args.target_dataset)
    train_dataset,test_dataset,linear_dataset = load_dataset(args.pretrain,args.target_dataset,args.surrogate_dataset,0,args.split)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        drop_last=True,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=True,
        drop_last=True,
    )
    linear_loader = torch.utils.data.DataLoader(
        linear_dataset,
        batch_size=32,
        shuffle=True,
        drop_last=True,
    )
    test_for_target(target_encoder, target_linear, test_loader)
    if(args.steal=='label'):
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(surrogate_model.parameters(), lr=3e-4)
        for i in range(100):
            train_onehot(target_encoder,target_linear,surrogate_model,train_loader,criterion,optimizer)
            agreement,accuracy = test_onehot(target_encoder,target_linear,surrogate_model,test_loader)

    if(args.steal=='representation'):
        crtierion1 = torch.nn.MSELoss()
        criterion2 = torch.nn.CrossEntropyLoss()
        optimizer1 = torch.optim.Adam(surrogate_model.encoder.parameters(), lr=3e-4)
        optimizer2 = torch.optim.Adam(surrogate_model.linear.parameters(), lr=3e-4)
        for i in range(args.epoch):
            train_representation(target_encoder,target_linear,surrogate_model.encoder,train_loader,crtierion1,optimizer1,device)
            os.makedirs("normal_target_model/", exist_ok=True)
            if(i % 20 == 0):
                torch.save(surrogate_model.state_dict(), 'normal_target_model/'+args.model_type + '_' + args.target_dataset + '_' + args.surrogate_dataset + '_' + args.steal + '_' + args.surrogate_model + '_' + str(i) +'.pkl')
        for i in range(100):
            train_represnetation_linear(surrogate_model,target_encoder,target_linear,linear_loader,criterion2,optimizer2,device)
            agreement,accuracy = test_onehot(target_encoder,target_linear,surrogate_model,test_loader)
    if(args.steal=='posterior'):
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(surrogate_model.parameters(), lr=3e-4)
        for i in range(args.epoch):
            train_posterior(target_encoder,target_linear,surrogate_model,train_loader,criterion,optimizer)
            agreement,accuracy = test_onehot(target_encoder,target_linear,surrogate_model,test_loader)
    # torch.save(surrogate_model,'new_surrogate_model/surrogate_model_'+args.steal+'_' + args.model_type + '_'+args.pretrain+'_'+args.target_dataset+'_'+args.surrogate_dataset+'_' +args.args.surrogate_model +'.pkl')

if __name__ == "__main__":
    main()
    
