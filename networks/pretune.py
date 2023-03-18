import os
import sys
sys.path.append('/home/zhicai/bivaegan/TZSL/')
from turtle import forward
import numpy as np
from  torch.autograd import Variable 
from torch import autograd
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import torch.nn.functional as F
from datasets import ZSLDataset as util
from visual import tsne_visual
from networks.VAEGANV1_model import MLP_CRITIC,MLP_CRITIC_un,netRCritic
import warnings

warnings.filterwarnings('ignore')

import logging
logger = logging.getLogger("train")

class norm(nn.Module):
    def __init__(self,radius=1):
        super().__init__()
        self.radius = radius
    def forward(self,x):
        x = self.radius * x/ torch.norm(x,p=2,dim=-1,keepdim=True)
        return x  

class cls(nn.Module):
    def __init__(self,opt,nclass) -> None:
        super().__init__()
        self.cls_seen = nn.Sequential(*[nn.Linear(opt.t_dim,512),
                                        nn.BatchNorm1d(512),
                                        nn.LeakyReLU(0.2,True),
                                        nn.Linear(512,nclass)])
        self.seen_criterion = nn.CrossEntropyLoss()
        self.cls_weight= 1
    def forward(self,x,label):
        predicts = self.cls_seen(x)
        cls_loss = self.seen_criterion(predicts,label)
        loss =self.cls_weight *cls_loss
        return loss
    def inference(self,x):
        predicts = self.cls_seen(x)
        return predicts
    def forward(self, x):
        h = self.lrelu(self.fc1(x))
        h = self.fc2(h)
        return h
  


class t_cls(nn.Module):
    def __init__(self,opt,nclass) -> None:
        super().__init__()
        self.MLP_E=nn.Sequential(*[nn.Linear(2048,4096),
                                nn.BatchNorm1d(4096),
                                nn.LeakyReLU(0.2,True),
                                nn.Linear(4096,opt.t_dim),
                                nn.ReLU()
                                ])
        self.cls_seen = nn.Sequential(*[nn.Linear(opt.t_dim,512),
                                        nn.BatchNorm1d(512),
                                        nn.LeakyReLU(0.2,True),
                                        nn.Linear(512,nclass)])

        self.map_att =   nn.Sequential(*[nn.Linear(opt.t_dim,512),
                                        nn.LeakyReLU(0.2,True),
                                        nn.Linear(512,opt.attSize),
                                        norm(opt.att_radius)])


        self.MLP_D=nn.Sequential(*[nn.Linear(opt.t_dim,4096),
                                nn.BatchNorm1d(4096),
                                nn.LeakyReLU(0.2,True),
                                nn.Linear(4096,2048),
                                ])

        self.seen_criterion = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()
        self.recon_weight = 1
        self.bmn_weight = 0
        self.cls_weight = 20
        self.recon_att_weight = 1
        self.norm = opt.L2_norm
        self.radius = opt.radius
        if opt.att_criterian =='W1':    
            self.att_criterian = self.reconstruct_W1_loss
        else :
            self.att_criterian = self.reconstruct_mse_loss
        
    def _forward(self,x):
        hidden = self.MLP_E(x)
        recon_x = self.MLP_D(hidden)
        return hidden,recon_x

    def reconstruct_mse_loss(self,h_att,att):
        mse = self.mse(h_att,att)
        mse = mse.sum()/att.size(0)
        return mse

    def reconstruct_W1_loss(self,h_att,att):
        wt = (h_att-att).pow(2)
        wt /= wt.sum(1).sqrt().unsqueeze(1).expand(wt.size(0),wt.size(1))
        loss = wt * (h_att-att).abs()
        return loss.sum()/loss.size(0)
    def get_fake_att(self,x):
        h = self.MLP_E(x)
        map_a = self.map_att(h)
        return map_a
    
    def forward_seen(self,x,label = None ,att = None ,reconstruct=False):
        hidden,recon_x = self._forward(x)
        loss = 0
        if reconstruct:
            recon_loss = self.mse(x,recon_x).sum()/x.size(0)
            loss += self.recon_weight * recon_loss 

        if label is not None:
            predicts = self.cls_seen(hidden)
            cls_loss = self.seen_criterion(predicts,label)
            loss +=self.cls_weight *cls_loss
        if att  is not None:
            mapped_att = self.map_att(hidden)
            att_loss = self.att_criterian(mapped_att,att)
            loss +=self.recon_att_weight * att_loss

        return recon_x,loss
    
    def inference_seen(self,x):
        hidden,recon_x = self._forward(x)
        predicts = self.cls_seen(hidden)
        return predicts
    
    def forward_unseen(self,x):
        hidden,recon_x = self._forward(x)
        recon_loss = self.mse(x,recon_x).sum()/x.size(0)
        loss = self.recon_weight * recon_loss 
        return loss



def calc_gradient_penalty(opt,netD,real_data, fake_data, input_att = None,lambda1 = 1):
    alpha = torch.rand(opt.batch_size, 1)
    alpha = alpha.expand(real_data.size())
    if opt.cuda:
        alpha = alpha.cuda()
    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates = torch.norm(real_data,p=2,dim=1,keepdim=True) * interpolates / torch.norm(interpolates,p=2,dim=1,keepdim=True)
    if opt.cuda:
        interpolates = interpolates.cuda()
    interpolates = Variable(interpolates, requires_grad=True)
    if input_att is not None:
        disc_interpolates = netD(interpolates, Variable(input_att))
    else:
        disc_interpolates = netD(interpolates)
    ones = torch.ones(disc_interpolates.size())
    if opt.cuda:
        ones = ones.cuda()
    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=ones,
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda1
    return gradient_penalty


def pretune(opt,data,show_visual=False,save = False):
    opt.t_dim = 2048
    net = t_cls(opt,nclass=data.ntrain_class).cuda()     
    optimizer = optim.Adam(net.parameters(), lr=0.0001, betas=(0.5, 0.999))
    Da = netRCritic(opt).cuda()
    optimizer_Da = optim.Adam(Da.parameters(), lr=0.0001, betas=(0.5, 0.999))
    inputv = torch.FloatTensor(opt.batch_size, opt.resSize).cuda()
    inputa = torch.FloatTensor(opt.batch_size, opt.attSize).cuda()
    labelv = torch.LongTensor(opt.batch_size).cuda()
    bestAcc = 0
    Vars = []
    Distances = []
    use_Da = False
    reconstruct = False
    for epoch in range(opt.tune_epoch):
        
        
        for i in range(0, data.ntrain, opt.batch_size):
            net.zero_grad()
            batch_input, batch_att, batch_label= data.next_seen_batch(opt.batch_size, return_mapped_label = True) 
            inputv.copy_(batch_input)
            inputa.copy_(batch_att[0])
            labelv.copy_(batch_label)
            reconx,loss1 = net.forward_seen(inputv,att=inputa , label=labelv,reconstruct= reconstruct)
            loss1.backward()
            optimizer.step()
            
        if reconstruct:
            for i in range(0,data.ntest_unseen,opt.batch_size):
                if use_Da:
                    net.eval()
                    Da.train()

                    for _ in range(3):
                        Da.zero_grad()
                        batch_input,batch_ran_att, = data.next_unseen_batch(opt.batch_size)
                        inputv.copy_(batch_input)  
                        inputa.copy_(batch_ran_att)
                        real = -100 * Da(inputa).mean()
                        fake_data = net.get_fake_att(inputv)
                        fake = 100* Da(fake_data).mean()
                        gp = calc_gradient_penalty(opt,Da,inputa,fake_data,lambda1=10)
                        loss = real + fake + gp
                        loss.backward()
                        optimizer_Da.step()
                    net.train()
                    Da.eval()
                net.zero_grad()

                inputv.copy_(batch_input)   
                batch_input,_, = data.next_unseen_batch(opt.batch_size)
                if use_Da:
                    loss1 = -1 * Da(net.get_fake_att(inputv)).mean()
                    loss1.backward()
                
                loss2 = net.forward_unseen(inputv)
                loss2.backward()
                optimizer.step()

        test_X = data.test_seen_feature
        test_label = data.test_seen_mapped_label

        start = 0
        ntest = test_X.size()[0]
        predicted_label = torch.LongTensor(test_label.size())
        net.eval()
        for i in range(0, ntest, opt.batch_size):
            end = min(ntest, start+opt.batch_size)
            inputX = Variable(test_X[start:end], volatile=True).cuda()
            predicts =net.inference_seen(inputX)  
            _, predicted_label[start:end] = torch.max(predicts.data, 1)
            start = end
        acc = torch.sum(predicted_label==test_label)/ntest
        acc = acc.item()
        if acc > bestAcc:
            bestAcc = acc
        print(f"acc: {np.round(acc,5)}")

        tune_test_unseen_feature=[]
        i = 0
        while i*opt.batch_size < data.ntest_unseen:
            inputRes = data.test_unseen_feature[i*opt.batch_size:(i+1)*opt.batch_size]
            tuneRes,_ = net._forward(inputRes.cuda())
            tune_test_unseen_feature.append(tuneRes.detach().cpu())
            i+=1
        tune_test_unseen_feature=torch.cat(tune_test_unseen_feature,dim=0)
        unseen_meanStatics=[]
        unseen_varStatics=[]
        for i, idx in enumerate(data.unseen_classes_dict):
            unseen_meanStatics.append(torch.mean(tune_test_unseen_feature[idx],dim=0))
            unseen_varStatics.append(torch.var(tune_test_unseen_feature[idx],dim=0))
        unseen_meanStatics=torch.stack(unseen_meanStatics,dim=0)
        unseen_varStatics=torch.stack(unseen_varStatics,dim=0)
        Var = torch.norm(unseen_varStatics,dim=1).mean()
        _,sort,d = cal_sim(unseen_meanStatics) 
        distance = torch.sum(d,dim=1)/(d.size(0)-1)
        Vars.append(Var)
        Distances.append(distance.mean())
        print(f'var: {Var}  dis: {distance.mean()}')
        logger.info(f'var: {Var}  dis: {distance.mean()}')
        net.train()
    logger.info(f"best acc: {bestAcc}\n")
    # print(Vars)
    # print(Distances)
    # print("use_tuned_feature!!\n")
    net.eval()

    # if opt.use_ajust_att:
    #     ajusted_att= net.ajustatt(data.attribute.cuda()).detach().cpu()
    #     # data.attribute = ajusted_att
    #     data.attribute = torch.cat([data.attribute,ajusted_att],dim=1) 
    #     data.attribute = norm(1)(data.attribute)
    #     opt.attSize = 2 * opt.attSize


    if show_visual:
        tune_test_unseen_feature=[]
        i = 0
        tsne_visual(np.array(data.test_unseen_feature),np.array(data.test_unseen_label),path = f'{opt.dataset}_{bestAcc}_untuned.pdf')
        while i*opt.batch_size < data.ntest_unseen:
            inputRes = data.test_unseen_feature[i*opt.batch_size:(i+1)*opt.batch_size]
            tuneRes,_ = net._forward(inputRes.cuda())
            tune_test_unseen_feature.append(tuneRes.detach().cpu())
            i+=1
        tune_test_unseen_feature=torch.cat(tune_test_unseen_feature,dim=0)
        logger.info(len(tune_test_unseen_feature))
        tsne_visual(np.array(tune_test_unseen_feature),np.array(data.test_unseen_label),path = f'{opt.dataset}_E{opt.tune_epoch}_{bestAcc}_tuned_.pdf')

    if opt.pretune_feature:
        i = 0
        tune_train_feature=[]
        while i*opt.batch_size < data.ntrain:
            inputRes = data.train_feature[i*opt.batch_size:(i+1)*opt.batch_size]
            tuneRes,_ = net._forward(inputRes.cuda())
            tune_train_feature.append(tuneRes.detach().cpu())
            i+=1
        tune_train_feature=torch.cat(tune_train_feature,dim=0)
        logger.info(len(tune_train_feature))
        data.train_feature=tune_train_feature

        i = 0
        tune_test_seen_feature=[]
        while i*opt.batch_size < data.ntest_seen:
            inputRes = data.test_seen_feature[i*opt.batch_size:(i+1)*opt.batch_size]
            tuneRes,_ = net._forward(inputRes.cuda())
            tune_test_seen_feature.append(tuneRes.detach().cpu())
            i+=1
        tune_test_seen_feature=torch.cat(tune_test_seen_feature,dim=0)
        logger.info(len(tune_test_seen_feature))
        data.test_seen_feature=tune_test_seen_feature

        i = 0
        tune_test_unseen_feature=[]
        while i*opt.batch_size < data.ntest_unseen:
            inputRes = data.test_unseen_feature[i*opt.batch_size:(i+1)*opt.batch_size]
            tuneRes,_ = net._forward(inputRes.cuda())
            tune_test_unseen_feature.append(tuneRes.detach().cpu())
            i+=1
        tune_test_unseen_feature=torch.cat(tune_test_unseen_feature,dim=0)
        logger.info(len(tune_test_unseen_feature))
        data.test_unseen_feature=tune_test_unseen_feature

        data.norm_feature()
        opt.resSize=opt.t_dim
        

        if save:
            if opt.L2_norm:
                if os.path.exists(f'datasets/{opt.dataset}'):
                    pass
                else:
                    os.mkdir(f'datasets/{opt.dataset}')
                new_feature_path = f'datasets/{opt.dataset}/{opt.tag}_Epo{opt.tune_epoch}_acc{np.round(bestAcc,4)}_r{opt.radius}.pth'
               
                data.save_feature(new_feature_path)

def cal_sim(data):
    sim = data @ data.transpose(0,1)
    sim_sort = torch.argsort(sim,dim=1,descending=True)
    data_d = data.unsqueeze(0) - data.unsqueeze(1)
    data_d = torch.sqrt(torch.square(data_d).sum(dim=2))
    return sim,sim_sort[:10,:6],data_d




