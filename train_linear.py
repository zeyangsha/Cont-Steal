import torch
import torchvision
from torchvision import datasets
from tqdm import tqdm
import torch.nn as nn
from sklearn.metrics import accuracy_score
from utils import load_normal_model,load_dataset
from Linear import linear
from utils import load_target_model
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--model_type',default='byol',type=str)
parser.add_argument('--encoder_dataset',default='cifar10',type=str)
parser.add_argument('--classifier_dataset',default='cifar10',type=str)
parser.add_argument('--surrogate_dataset',default='cifar10',type=str)
parser.add_argument('--topk',default=512,type = int)
parser.add_argument('--round_size',default=0,type = int)
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
target_encoder,target_linear = load_target_model(args.model_type,args.encoder_dataset,args.classifier_dataset)
if(args.classifier_dataset == 'cifar100'):
    num = 100
else:
    num = 10
linear_model = linear(target_encoder.inplanes,num).to(device)
train_dataset,test_dataset,linear_dataset = load_dataset(args.classifier_dataset,args.classifier_dataset,args.surrogate_dataset,0,1)
train_loader = torch.utils.data.DataLoader(
        linear_dataset,
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
optimizer = torch.optim.Adam(linear_model.parameters(), lr=3e-4)
criterion = torch.nn.CrossEntropyLoss()
best_result = 0
best_static = 0

if(args.round_size == 0):
    for i in range(200):

        print(i)
        accuracy_epoch = 0
        loss_epoch = 0
        best_result = 0

        accuracy_sample = []
        total_sample = []

        for step, (x,y) in enumerate(tqdm(train_loader)):

            target_encoder.eval()
            target_encoder.requires_grad = False

            linear_model.train()
            x = x.cuda()
            y = y.cuda()
            re = target_encoder(x)
            output = linear_model(re)

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

        print(accuracy_score(total_sample,accuracy_sample))
        print(loss_epoch/len(train_loader))

        print("test_begin")
        test_accuracy_sample = []
        test_total_sample = []
        for step, (x, y) in enumerate(test_loader):
            x = x.cuda()
            y = y.cuda()
            target_encoder.eval()
            linear_model.eval()
            re = target_encoder(x)
            output = linear_model(re)
            predicted = output.argmax(1)
            predicted = predicted.cpu().numpy()
            predicted = list(predicted)
            test_accuracy_sample.extend(predicted)
            y = y.cpu().numpy()
            y = list(y)
            test_total_sample.extend(y)
        test_result = accuracy_score(test_total_sample, test_accuracy_sample)
        print(accuracy_score(test_total_sample, test_accuracy_sample))
        if(test_result > best_result):
            best_result = test_result
            os.makedirs("new_target_model/"+args.encoder_dataset+'/', exist_ok=True)
            print('new_target_model/'+args.encoder_dataset+'/'+args.model_type + '_' + args.classifier_dataset +'_linear.pkl')
            torch.save(linear_model.state_dict(), 'new_target_model/'+args.encoder_dataset+'/'+args.model_type + '_' + args.classifier_dataset +'_linear.pkl')
elif(args.topk == 512):
    for i in range(200):
    
        print(i)
        accuracy_epoch = 0
        loss_epoch = 0
        best_result = 0

        accuracy_sample = []
        total_sample = []

        for step, (x,y) in enumerate(tqdm(train_loader)):

            target_encoder.eval()
            target_encoder.requires_grad = False

            linear_model.train()
            x = x.cuda()
            y = y.cuda()
            r = target_encoder(x)
            n_digits = args.round_size
            r = torch.round(r * 10**n_digits) / 10**n_digits
            output = linear_model(r)

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

        print(accuracy_score(total_sample,accuracy_sample))
        print(loss_epoch/len(train_loader))

        print("test_begin")
        test_accuracy_sample = []
        test_total_sample = []
        for step, (x, y) in enumerate(test_loader):
            x = x.cuda()
            y = y.cuda()
            target_encoder.eval()
            linear_model.eval()
            re = target_encoder(x)
            output = linear_model(re)
            predicted = output.argmax(1)
            predicted = predicted.cpu().numpy()
            predicted = list(predicted)
            test_accuracy_sample.extend(predicted)
            y = y.cpu().numpy()
            y = list(y)
            test_total_sample.extend(y)
        test_result = accuracy_score(test_total_sample, test_accuracy_sample)
        print(accuracy_score(test_total_sample, test_accuracy_sample))
        if(test_result > best_result):
            best_result = test_result
            os.makedirs("new_target_model/"+args.encoder_dataset+'/', exist_ok=True)
            print('new_target_model/'+args.encoder_dataset+'/'+args.model_type + '_' + args.classifier_dataset +'_linear.pkl')
            torch.save(linear_model.state_dict(), 'new_target_model/'+args.encoder_dataset+'/'+args.model_type + '_' + args.classifier_dataset +'_linear.pkl')
        
