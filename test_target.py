import torch
from sklearn.metrics import accuracy_score

def test_for_target(target_encoder,target_linear,test_loader):
    print("test for target model")
    accuracy_sample = []
    total_sample = []
    for step,(x,y) in enumerate(test_loader):
        x = x.cuda()
        y = y.cuda()
        target_encoder.eval()
        target_linear.eval()
        re = target_encoder(x)
        output = target_linear(re)
        predicted = output.argmax(1)
        predicted = predicted.cpu().numpy()
        predicted = list(predicted)
        accuracy_sample.extend(predicted)
        y = y.cpu().numpy()
        y = list(y)
        total_sample.extend(y)
    print(accuracy_score(total_sample, accuracy_sample))
