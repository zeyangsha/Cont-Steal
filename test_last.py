from sklearn.metrics import accuracy_score
import torch.nn.functional as F
from tqdm import tqdm


def test_onehot(target_encoder,target_linear,surrogate_model,test_loader):
    print(" ")
    print("test for this epoch")
    surrogate_sample = []
    target_sample = []
    true_sample = []
    mse_epoch = 0
    for step, (x,y) in enumerate(tqdm(test_loader)):
        target_encoder.eval()
        target_linear.eval()
        surrogate_model.eval()
        x = x.cuda()
        y = y.cuda()

        re = target_encoder(x)
        output = target_linear(re)
        su_re = surrogate_model.encoder(x)
        mse = F.mse_loss(su_re,re)
        mse_epoch += mse.item()

        su_output = surrogate_model(x)

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
    print("test result")
    print("test agreement:")
    agreement = accuracy_score(target_sample,surrogate_sample)
    print(accuracy_score(target_sample,surrogate_sample))
    print("test accuracy:")
    accuracy = accuracy_score(true_sample,surrogate_sample)
    print(accuracy_score(true_sample,surrogate_sample))
    print("mse")
    print(mse_epoch/len(test_loader ) )
    return agreement,accuracy
