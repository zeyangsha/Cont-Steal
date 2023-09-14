import torch
import argparse
from surrogate_model import Surrogate_model
from utils import load_target_model,load_dataset
from Loss import ContrastiveLoss
from train_representation import train_representation,train_represnetation_linear,train_representation_msepluscontrastive
from test_last import test_onehot
from train_weight_onehot import train_confident_onehot,train_gap_onehot
import numpy as np
import dataloader
from test_target import test_for_target
import torchvision
from Linear import linear
import os

def main():
    torch.set_num_threads(1)   
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type',default='simclr',type=str)
    parser.add_argument('--target_dataset',default='cifar10',type=str)
    parser.add_argument('--surrogate_dataset',default='cifar100',type=str)
    parser.add_argument('--steal',default='mse+contrastive',type=str)
    parser.add_argument('--augmentation',default=0,type=int)
    parser.add_argument('--surrogate_model',default='resnet18',type=str)
    parser.add_argument('--epoch',default='0',type=int)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if(args.target_dataset == 'cifar10'):
        catagory_num = 10
    else:
        catagory_num = 100
    surrogate_model = Surrogate_model(catagory_num,args.surrogate_model).to(device)
    if(args.steal == 'mse+contrastive'):
        surrogate_model.load_state_dict(torch.load('new_target_model/'+args.model_type + '_' + args.target_dataset + '_' + args.surrogate_dataset + '_' + args.steal + '_' + args.surrogate_model + '_' + str(args.epoch) +'.pkl'))
    else:
        surrogate_model.load_state_dict(torch.load('normal_target_model/'+args.model_type + '_' + args.target_dataset + '_' + args.surrogate_dataset + '_' + args.steal + '_' + args.surrogate_model + '_' + str(args.epoch) +'.pkl'))
    target_encoder,target_linear = load_target_model(args.model_type,args.target_dataset)
    train_dataset,test_dataset,linear_dataset = load_dataset(args.target_dataset,args.surrogate_dataset,args.augmentation,1)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
        drop_last=True,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=128,
        shuffle=True,
        drop_last=True,
    )
    linear_loader = torch.utils.data.DataLoader(
        linear_dataset,
        batch_size=128,
        shuffle=True,
        drop_last=True,
    )
    criterion2 = torch.nn.CrossEntropyLoss()
    optimizer2 = torch.optim.Adam(surrogate_model.linear.parameters(), lr=3e-4)
    for i in range(100):
        train_represnetation_linear(surrogate_model,target_encoder,target_linear,linear_loader,criterion2,optimizer2,device)
        test_onehot(target_encoder,target_linear,surrogate_model,test_loader)

if __name__ == "__main__":
    main()