import torch
import torchvision.models as models
from Linear import linear
import torch.nn as nn
from torchvision import datasets
import torchvision
from target_model import target_model
from torchvision import transforms
from Linear import linear
from torchvision.transforms import transforms
from RandAugment import RandAugment
from torch.utils.data import random_split


class random_transform():

    def __init__(self,size):
        self.transform = torchvision.transforms.Compose(
            [
                transforms.Resize((size,size)),
                RandAugment(2, 14),
                transforms.RandomCrop(size, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )

    def __call__(self,x):
        return self.transform(x)

class easy_transform():
    
    def __init__(self, size):
        self.transform = torchvision.transforms.Compose(
            [
                transforms.Resize((size,size)),
                torchvision.transforms.RandomResizedCrop(size=size),
                torchvision.transforms.ToTensor(),
            ]
        )

    def __call__(self,x):
        return self.transform(x)

class normal_model(nn.Module):
    def __init__(self,out_dim):
        super(normal_model,self).__init__()
        self.encoder = torchvision.models.resnet18(pretrained=False)
        self.encoder.fc = nn.Identity()

        self.out_dim = out_dim
        self.linear = nn.Linear(512,self.out_dim)

    def forward(self,x):
        x = self.encoder(x)
        x = self.linear(x)
        return x

def load_normal_model(data_set):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if(data_set == 'cifar10' or data_set == 'stl10'):
        catagory_num = 10
    else:
        catagory_num = 100
    model = normal_model(catagory_num )
    model.load_state_dict(torch.load("target_model/"+data_set+"/normal_"+data_set+".pkl",map_location='cuda:0'))
    model = model.to(device)
    return model.encoder, model.linear

def new_load_target_model(model_type,pretrain,downstream):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18()
    model.fc = nn.Indentity()
    if(pretrain == 'cifar10'):
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2, bias=False)
        model.maxpool = nn.Identity()
    state = torch.load('target_downstream/'+pretrain+'/'+model_type+'_'+downstream+'.ckpt')["state_dict"]
    for k in list(state.keys()):
        if "backbone" in k:
            state[k.replace("backbone.", "")] = state[k]
        del state[k]
    model.load_state_dict(state, strict=False)
    model = model.to(device)
    linear_model = linear(model.inplanes,10)
    linear_state = torch.load("target_downstream/"+pretrain+"/"+model_type+'_'+downstream+'_linear.pkl')
    linear_model.load_state_dict(linear_state)
    linear_model = linear_model.to(device)
    return model,linear_model


def load_target_model(model_type,pretrain,downstream):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18()
    model.fc = nn.Identity()
    if(pretrain == 'cifar10'):
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2, bias=False)
        model.maxpool = nn.Identity()
    state = torch.load('target_downstream/'+pretrain+'/'+model_type+'_'+pretrain+'.ckpt')["state_dict"]
    for k in list(state.keys()):
        if "backbone" in k:
            state[k.replace("backbone.", "")] = state[k]
        del state[k]
    model.load_state_dict(state, strict=False)
    model = model.to(device)
    linear_model = linear(model.inplanes,10)
    linear_state = torch.load("target_downstream/"+pretrain+"/"+model_type+'_'+downstream+'_linear.pkl')
    linear_model.load_state_dict(linear_state)
    linear_model = linear_model.to(device)
    return model,linear_model

#def load_dataset(t_dataset,s_dataset,aug,eplision):
def load_dataset(pretrain ,t_dataset,s_dataset,aug,eplision):
    if(pretrain =='imagenet'):
        imagesize = 96
    else:
        imagesize = 32
    imagesize = 224
    if(aug == 0):
        if(s_dataset=='cifar10'):
            train_dataset = datasets.CIFAR10(root='data/cifar10_dataset', train=True,
                                        transform=torchvision.transforms.Compose([torchvision.transforms.Resize((imagesize,imagesize)),torchvision.transforms.ToTensor()]), download=True)
        if(s_dataset=='stl10'):
            train_dataset = datasets.STL10(root='data/stl10_dataset',split='train',transform=torchvision.transforms.Compose([torchvision.transforms.Resize((imagesize,imagesize)),torchvision.transforms.ToTensor()]),download=True)

        if(s_dataset=='cifar100'):
            print("yes")
            train_dataset = datasets.CIFAR100(root='data/cifar100_dataset', train=True,
                                        transform=torchvision.transforms.Compose([torchvision.transforms.Resize((imagesize,imagesize)),torchvision.transforms.ToTensor()]), download=True)
                  
        if(s_dataset == 'svhn'):
            train_dataset = datasets.SVHN('data/svhn_dataset', split='train',transform=torchvision.transforms.Compose([torchvision.transforms.Resize((imagesize,imagesize)),torchvision.transforms.ToTensor()]), download=True)

        if(s_dataset == 'mnist'):
            train_dataset = datasets.FashionMNIST('data/mnist_dataset',train=True, transform=torchvision.transforms.Compose([torchvision.transforms.Resize((imagesize,imagesize)),torchvision.transforms.Grayscale(3),torchvision.transforms.ToTensor(),]), download=True)

        if(t_dataset=='stl10'):
            test_dataset = datasets.STL10(root='data/stl10_dataset',split='test',transform=torchvision.transforms.Compose([torchvision.transforms.Resize((imagesize,imagesize)),torchvision.transforms.ToTensor()]),download=True)
            linear_dataset = datasets.STL10(root='data/stl10_dataset',split='train',transform=torchvision.transforms.Compose([torchvision.transforms.Resize((imagesize,imagesize)),torchvision.transforms.ToTensor()]),download=True)
        
        if(t_dataset=='cifar10'):
            test_dataset = datasets.CIFAR10(root='data/cifar10_dataset', train=False,
                                          transform=torchvision.transforms.Compose([torchvision.transforms.Resize((imagesize,imagesize)),torchvision.transforms.ToTensor()]), download=True)
            linear_dataset = datasets.CIFAR10(root='data/cifar10_dataset', train=True,
                                          transform=torchvision.transforms.Compose([torchvision.transforms.Resize((imagesize,imagesize)),torchvision.transforms.ToTensor()]), download=True)
        if(t_dataset=='cifar100'):
            test_dataset = datasets.CIFAR100(root='data/cifar100_dataset', train=False,
                                        transform=transforms.Compose([torchvision.transforms.ToTensor(),]), download=True)
            #test_dataset = datasets.CIFAR100(root='./data/cifar100_dataset', train=False,
            #                            transform=torchvision.transforms.ToTensor(), download=True)

            linear_dataset = datasets.CIFAR100(root='data/cifar100_dataset', train=True,
                                          transform=torchvision.transforms.ToTensor(), download=True)

        if(t_dataset=='svhn'):
            test_dataset = datasets.SVHN('data/svhn_dataset', split='test',transform=torchvision.transforms.Compose([torchvision.transforms.Resize((imagesize,imagesize)),torchvision.transforms.ToTensor()]), download=True)

            linear_dataset = datasets.SVHN('data/svhn_dataset', split='train',transform=torchvision.transforms.Compose([torchvision.transforms.Resize((imagesize,imagesize)),torchvision.transforms.ToTensor()]), download=True)
        
        if(t_dataset=='mnist'):
            test_dataset = datasets.FashionMNIST(root='data/mnist_dataset', train = False,
                                                transform=torchvision.transforms.Compose([torchvision.transforms.Resize((imagesize,imagesize)),torchvision.transforms.Grayscale(3),torchvision.transforms.ToTensor(),]), download=True)
            linear_dataset =  datasets.FashionMNIST(root='data/mnist_dataset', train = True,
                                                transform=torchvision.transforms.Compose([torchvision.transforms.Resize((imagesize,imagesize)),torchvision.transforms.Grayscale(3),torchvision.transforms.ToTensor(),]), download=True)

    elif(aug == 1):
       
        if(s_dataset=='stl10'):
            train_dataset = datasets.STL10(root='data/stl10_dataset',split='train+unlabeled',transform=easy_transform(imagesize),download=True)

        if(s_dataset=='cifar10'):
            train_dataset = datasets.CIFAR10(root='data/cifar10_dataset', train=True,
                                        transform=easy_transform(imagesize), download=True)
    
        if(s_dataset=='cifar100'):
            train_dataset = datasets.CIFAR100(root='data/cifar100_dataset', train=True,
                                        transform=easy_transform(imagesize), download=True)

        if(t_dataset=='stl10'):
            test_dataset = datasets.STL10(root='data/stl10_dataset',split='test',transform=torchvision.transforms.ToTensor(),download=True)
            linear_dataset = datasets.STL10(root='data/stl10_dataset',split='train',transform=torchvision.transforms.ToTensor(),download=True)
        
        if(t_dataset=='cifar10'):
            test_dataset = datasets.CIFAR10(root='data/cifar10_dataset', train=False,
                                        transform=torchvision.transforms.ToTensor(), download=True)
            linear_dataset = datasets.CIFAR10(root='data/cifar10_dataset', train=True,
                                          transform=torchvision.transforms.ToTensor(), download=True)

        if(t_dataset=='cifar100'):
            test_dataset = datasets.CIFAR100(root='data/cifar100_dataset', train=False,
                                        transform=torchvision.transforms.ToTensor(), download=True)
            linear_dataset = datasets.CIFAR100(root='data/cifar100_dataset', train=True,
                                          transform=torchvision.transforms.ToTensor(), download=True)

    elif(aug == 2):
        if(s_dataset=='stl10'):
            train_dataset = datasets.STL10(root='data/stl10_dataset',split='train+unlabeled',transform=random_transform(imagesize),download=True)

        if(s_dataset=='cifar10'):
            train_dataset = datasets.CIFAR10(root='data/cifar10_dataset', train=True,
                                        transform=random_transform(imagesize), download=True)
    
        if(s_dataset=='cifar100'):
            train_dataset = datasets.CIFAR100(root='data/cifar100_dataset', train=True,
                                        transform=random_transform(imagesize), download=True)

        if(s_dataset=='svhn'):
            train_dataset = datasets.SVHN('data/svhn_dataset', split='train',transform=random_transform(imagesize), download=True)
        
        if(s_dataset=='mnist'):
            train_dataset = datasets.FashionMNIST(root='data/mnist_dataset', train = True,
                                                transform=torchvision.transforms.Compose([torchvision.transforms.Grayscale(3),random_transform(imagesize)]), download=True)
        
        if(t_dataset=='cifar10'):
            test_dataset = datasets.CIFAR10(root='data/cifar10_dataset', train=False,
                                        transform=torchvision.transforms.Compose([torchvision.transforms.Resize((imagesize,imagesize)),torchvision.transforms.ToTensor()]), download=True)
            linear_dataset = datasets.CIFAR10(root='data/cifar10_dataset', train=True,
                                          transform=torchvision.transforms.Compose([torchvision.transforms.Resize((imagesize,imagesize)),torchvision.transforms.ToTensor()]), download=True)

        if(t_dataset=='cifar100'):
            test_dataset = datasets.CIFAR100(root='data/cifar100_dataset', train=False,
                                        transform=torchvision.transforms.Compose([torchvision.transforms.Resize((imagesize,imagesize)),torchvision.transforms.ToTensor()]), download=True)
            linear_dataset = datasets.CIFAR100(root='data/cifar100_dataset', train=True,
                                          transform=torchvision.transforms.Compose([torchvision.transforms.Resize((imagesize,imagesize)),torchvision.transforms.ToTensor()]), download=True)
        if(t_dataset=='stl10'):
            test_dataset = datasets.STL10(root='data/stl10_dataset',split='test',transform=torchvision.transforms.Compose([torchvision.transforms.Resize((imagesize,imagesize)),torchvision.transforms.ToTensor()]),download=True)
            linear_dataset = datasets.STL10(root='data/stl10_dataset',split='train',transform=torchvision.transforms.Compose([torchvision.transforms.Resize((imagesize,imagesize)),torchvision.transforms.ToTensor()]),download=True)

        if(t_dataset=='imagenet100'):
            test_dataset = datasets.ImageFolder('data/imagenet-100/val.X',transform=random_transform(224))
            linear_dataset = datasets.ImageFolder('data/imagenet-100/train',transform=random_transform(224))

        if(t_dataset=='svhn'):
            test_dataset = datasets.SVHN('data/svhn_dataset', split='test',transform=torchvision.transforms.Compose([torchvision.transforms.Resize((imagesize,imagesize)),torchvision.transforms.ToTensor()]), download=True)

            linear_dataset = datasets.SVHN('data/svhn_dataset', split='train',transform=torchvision.transforms.Compose([torchvision.transforms.Resize((imagesize,imagesize)),torchvision.transforms.ToTensor()]), download=True)
        
        if(t_dataset=='mnist'):
            test_dataset = datasets.FashionMNIST(root='data/mnist_dataset', train = False,
                                                transform=torchvision.transforms.Compose([torchvision.transforms.Resize((32,32)),torchvision.transforms.Grayscale(3),torchvision.transforms.ToTensor(),]), download=True)
            linear_dataset = datasets.FashionMNIST(root='data/mnist_dataset', train = True,
                                                transform=torchvision.transforms.Compose([torchvision.transforms.Resize((32,32)),torchvision.transforms.Grayscale(3),torchvision.transforms.ToTensor(),]), download=True)


    size = len(train_dataset)
    newsize = size*eplision
    
    train_dataset,_ = random_split(dataset=train_dataset,lengths=[int(newsize),int(size-newsize)],generator=torch.Generator().manual_seed(0))
    print(len(train_dataset))
    return train_dataset,test_dataset,linear_dataset
                                    