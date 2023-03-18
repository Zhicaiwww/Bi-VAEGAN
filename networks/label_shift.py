from pickletools import optimize
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
#---------------------- utility functions used ----------------------------
def idx2onehot(a,k):
    a=a.astype(int)
    b = np.zeros((a.size, k))
    b[np.arange(a.size), a] = 1
    return b


def confusion_matrix(ytrue, ypred,k):
    # C[i,j] denotes the frequency of ypred = i, ytrue = j.
    n = ytrue.size
    C = np.dot(idx2onehot(ypred,k).T,idx2onehot(ytrue,k))
    return C/n

def confusion_matrix_probabilistic(ytrue, ypred,k):
    # Input is probabilistic classifiers in forms of n by k matrices
    n,d = np.shape(ypred)
    C = np.dot(ypred.T, idx2onehot(ytrue,k))
    return C/n


def calculate_marginal(y,k):
    mu = np.zeros(shape=(k,1))
    for i in range(k):
        mu[i] = np.count_nonzero(y == i)
    return mu/np.size(y)

def calculate_marginal_probabilistic(y,k):
    return np.mean(y,axis=0)

def estimate_labelshift_ratio(ytrue_s, soft_predict, soft_predict_t,k):
    if soft_predict.ndim == 2: # this indicates that it is probabilistic
        C = confusion_matrix_probabilistic(ytrue_s,soft_predict,k)
        mu_t = calculate_marginal_probabilistic(soft_predict_t, k)
    else:
        C = confusion_matrix(ytrue_s, soft_predict,k)
        mu_t = calculate_marginal(soft_predict_t, k)
    lamb = (1/min(len(soft_predict),len(soft_predict_t)))
    eigs = np.linalg.eigvals(C)
    print(eigs)
    print(np.sort(eigs))
    wt = np.linalg.solve(np.dot(C.T, C)+lamb*np.eye(k), np.dot(C.T, mu_t))
    return wt

def smooth_frequency(q,gamma=1):
    q = q / q.sum()
    q = np.power(q,gamma)
    q = q/ q.sum()
    return q
class LINEAR_LOGSOFTMAX_CLASSIFIER(nn.Module):
    def __init__(self, input_dim, nclass):
        super(LINEAR_LOGSOFTMAX_CLASSIFIER, self).__init__()
        # self.mlp = nn.Sequential(*[nn.Linear(input_dim,512),
        #                            nn.ReLU(),
        #                            nn.Linear(512,nclass)])
        self.fc = nn.Linear(input_dim, nclass)
        self.logic = nn.Softmax(dim=1)
    def forward(self, x): 
        o = self.logic(self.fc(x))
        self.o = o
        out = torch.log(o)
        return out
    def get_logic(self):
        return self.o
    def _init_weights(self,):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            
    
    
    

class ls(nn.Module):
    def __init__(self,synX,synY,synX2,synY2,testX,_batch_size=64,nepoch=20,nclass=10,att_size=85,netR=None,soft = True):
        super().__init__()
        self.epochs_completed = 0
        self.index_in_epoch=0
        self.nclass=nclass
        self.nepoch=nepoch
        self.batch_size=_batch_size
        self.ntrain,self.dim = synX.size()
        self.Xtrain = synX
        self.ytrain = synY
        self.valX = synX2
        self.valY = synY2
        self.testX = testX
        if netR is not None:
            self.netR = netR
            self.netR.eval()
            self.dim = att_size + self.dim + 4096
            self.Xtrain = self.compute_dec_out(self.Xtrain, self.dim)
            self.valX = self.compute_dec_out(self.valX, self.dim)
            self.testX = self.compute_dec_out(self.testX, self.dim)

        self.input = torch.FloatTensor(_batch_size, self.dim) 
        self.label = torch.LongTensor(_batch_size) 
        self.criterion = nn.NLLLoss()
        self.cuda=False
        self.model = LINEAR_LOGSOFTMAX_CLASSIFIER(self.dim,nclass=self.nclass)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, betas=(0.5, 0.999))
        self.soft = soft 
        
    def compute_dec_out(self, test_X, new_size):
        start = 0
        ntest = test_X.size()[0]
        new_test_X = torch.zeros(ntest,new_size)
        for i in range(0, ntest, self.batch_size):
            end = min(ntest, start+self.batch_size)
            if self.cuda:
                inputX = Variable(test_X[start:end].cuda(), volatile=True)
            else:
                inputX = Variable(test_X[start:end], volatile=True)
            feat1 = self.netR(inputX)
            feat2 = self.netR.getLayersOutDet()
            # feat2 = self.netR.getLayersOutDet()
            new_test_X[start:end] = torch.cat([inputX,feat1,feat2],dim=1).data.cpu()
            # new_test_X[start:end] = feat1.data.cpu()
            start = end
        return new_test_X

    def next_batch(self, batch_size):

        start = self.index_in_epoch
        # shuffle the data at the first epoch
        if self.epochs_completed == 0 and start == 0:
            perm = torch.randperm(self.ntrain)
            self.Xtrain = self.Xtrain[perm]
            self.ytrain = self.ytrain[perm]
        # the last batch
        if start + batch_size > self.ntrain:
            self.epochs_completed += 1
            rest_num_examples = self.ntrain - start
            if rest_num_examples > 0:
                X_rest_part = self.Xtrain[start:self.ntrain]
                Y_rest_part = self.ytrain[start:self.ntrain]
            # shuffle the data
            perm = torch.randperm(self.ntrain)
            self.Xtrain = self.Xtrain[perm]
            self.ytrain = self.ytrain[perm]
            # start next epoch
            start = 0
            self.index_in_epoch = batch_size - rest_num_examples
            end = self.index_in_epoch
            X_new_part = self.Xtrain[start:end]
            Y_new_part = self.ytrain[start:end]
            #print(start, end)
            if rest_num_examples > 0:
                return torch.cat((X_rest_part, X_new_part), 0) , torch.cat((Y_rest_part, Y_new_part), 0)
            else:
                return X_new_part, Y_new_part
        else:
            self.index_in_epoch += batch_size
            end = self.index_in_epoch
            return self.Xtrain[start:end], self.ytrain[start:end]
        
    def predict_wt(self):
        ntrain = self.Xtrain.size(0)
        self.model._init_weights()

        for epoch in range(self.nepoch):
            for i in range(0, ntrain, self.batch_size): 
                self.model.zero_grad()
                batch_input, batch_label = self.next_batch(self.batch_size) 
                self.input.copy_(batch_input)
                self.label.copy_(batch_label)
                inputv = Variable(self.input)
                labelv = Variable(self.label)
                output = self.model(inputv)
                loss = self.criterion(output, labelv)
                loss.backward()
                self.optimizer.step()
                
        hard_predict,soft_predict,freq = self.val(self.valX,ylabel=self.valY)
        hard_predict_t,soft_predict_t,freq_s = self.val(self.testX)
        print(f'ls frequency:{freq_s}')
        c, b = torch.max(soft_predict_t,dim=1,keepdim=True)
        confidence = torch.zeros_like(soft_predict_t).scatter_(1,b,c).mean(0)
        print(f'confidence:{confidence}')
        if self.soft:
            w = self.get_w(soft_predict,soft_predict_t)
        else:
            w = self.get_w(hard_predict,hard_predict_t)
        return w
    
    # def finetune(self,data,data_y,val,val_y):
    #     input_size = data.size(0)
    #     input_size_val = val.size(0)
    #     perm = torch.randperm(self.ntrain)
    #     self.Xtrain = self.Xtrain[perm]
    #     self.ytrain = self.ytrain[perm]
    #     self.Xtrain = torch.cat([self.Xtrain[input_size:],data],dim=0) 
    #     self.ytrain = torch.cat([self.ytrain[input_size:],data_y],dim=0) 
    #     self.valX = torch.cat([self.valX[input_size_val:],val],dim=0) 
    #     self.Y = torch.cat([self.va l[input_size_val:],val_y],dim=0) 
    #     return self.predict_wt()
        
    def get_w(self,hard_predict,hard_predict_t):
        w = estimate_labelshift_ratio(np.array(self.valY), np.array(hard_predict.detach()), np.array(hard_predict_t.detach()),self.nclass)
        return w
        
    def val(self,valX,ylabel=None):
        start = 0
        ntest = valX.size()[0]
        predicted_label = torch.LongTensor(valX.size(0))
        soft_label = torch.FloatTensor(valX.size(0),self.nclass)
        for i in range(0, ntest, self.batch_size):
            end = min(ntest, start+self.batch_size)
            if self.cuda:
                inputX = Variable(valX[start:end].cuda(), volatile=True)
            else:
                inputX = Variable(valX[start:end], volatile=True)
            output = self.model(inputX)
            softout = self.model.get_logic()
            # frequency += output.detach().cpu().sum(0)
            _, predicted_label[start:end] = torch.max(output.data, 1)
            soft_label[start:end]=softout
            start = end
        frequency = predicted_label.bincount()/valX.size(0)
        frequency = frequency/frequency.sum()
        return predicted_label,soft_label,frequency
        # frequency = predicted_label.bincount()/len(self.valY)
        # frequency = frequency/frequency.sum()

        