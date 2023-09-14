import torch
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
import numpy as np

def train_representation(target_encoder,surrogate_model,train_loader,criterion,optimizer,device):
    loss_epoch = 0
    mse_epoch = 0
    for step, (x,y) in enumerate(train_loader):
        target_encoder.eval()

        surrogate_model.train()
        optimizer.zero_grad()
        target_encoder.requires_grad = False
        x = x.to(device)
        y = y.to(device)
        re = target_encoder(x)
        su_output = surrogate_model(x)
        loss = criterion(su_output,re)
        loss.backward()
        optimizer.step()
        loss_epoch += loss.item()
        mse = F.mse_loss(su_output,re)
        mse_epoch += mse.item()
    print("loss")
    print(loss_epoch/len(train_loader))
    print("mse")
    print(mse_epoch/len(train_loader))
    print("")

def train_represnetation_linear(model,target_encoder,train_loader,criterion,optimizer,device):
    accuracy_sample = []
    total_sample = []
    target_sample = []
    loss_epoch = 0
    for step, (x,y) in enumerate(train_loader):
        model.encoder.requires_grad = False
        model.encoder.eval()
        optimizer.zero_grad()
        x = x.to(device)
        y = y.to(device)
        model.linear.train()
        re = model.encoder(x)
        output = model.linear(re)
        loss = criterion(output, y)
        predicted = output.argmax(1)
        predicted = predicted.cpu().numpy()
        predicted = list(predicted)
        accuracy_sample.extend(predicted)
        y = y.cpu().numpy()
        y = list(y)
        total_sample.extend(y)
        loss.backward()
        optimizer.step()
        loss_epoch += loss.item()
        target_linear.eval()
        target_encoder.eval()
        t_output = target_linear(target_encoder(x))
        t_predicted = t_output.argmax(1)
        t_predicted = t_predicted.cpu().numpy()
        t_predicted = list(t_predicted)
        target_sample.extend(t_predicted)
    print("accuracy:")
    print(accuracy_score(total_sample, accuracy_sample))
    print("agreement:")
    print(accuracy_score(target_sample,accuracy_sample))
    print("loss")
    print(loss_epoch / len(train_loader))
