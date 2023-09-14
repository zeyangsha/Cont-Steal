import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import accuracy_score

def train_posterior(target_encoder,target_linear,surrogate_model,train_loader,criterion,optimizer):
    surrogate_sample = []
    target_sample = []
    true_sample = []
    mse_epoch = 0
    for step, (x,y) in enumerate(train_loader):
        target_encoder.eval()
        target_linear.eval()
        surrogate_model.train()
        optimizer.zero_grad()
        target_encoder.requires_grad = False
        target_linear.requires_grad = False
        x = x.cuda()
        y = y.cuda()
        re = target_encoder(x)
        output = target_linear(re)
        su_output = surrogate_model(x)
        su_re = surrogate_model.encoder(x)
        mse = F.mse_loss(su_re,re)
        mse_epoch += mse.item()
        loss = criterion(su_output,output)
        loss.backward()
        optimizer.step()
        ta_predicted = output.argmax(1)
        ta_predicted = ta_predicted.cpu().numpy()
        ta_predicted = list(ta_predicted)
        target_sample.extend(ta_predicted)
        su_output = su_output.argmax(1)
        su_output = su_output.cpu().numpy()
        su_output = list(su_output)
        surrogate_sample.extend(su_output)
        y = y.cpu().numpy()
        y = list(y)
        true_sample.extend(y)

    print("agreement:")
    print(accuracy_score(target_sample,surrogate_sample))
    print(" ")
    print("accuracy:")
    print(accuracy_score(true_sample,surrogate_sample))
