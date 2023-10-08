#import h5py
import numpy as np
from requests import patch
import scipy.io as sio
import torch
import torch.nn.functional as F
from sklearn import preprocessing
import logging
import pandas as pd
import torch.nn as nn
import sys
sys.path.append('/home/zhicai/tfvaegan-master/datasets/')
# from FSLDataset import _load_pickle,_datasetTrainFeaturesFiles,_datasetFeaturesFiles

logger = logging.getLogger("train")

def perb_att(input_att):
    gama = torch.rand(input_att.size())
    noise = 0.1 * gama * input_att
    att = norm(1)(noise+input_att)
    return att

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def map_label(label, classes):
    mapped_label = torch.LongTensor(label.size())
    for i in range(classes.size(0)):
        mapped_label[label==classes[i]] = i    

    return mapped_label


class norm(nn.Module):
    def __init__(self,radius=1):
        super().__init__()
        self.radius = radius
    def forward(self,x):
        x = self.radius * x/ torch.norm(x,p=2,dim=-1,keepdim=True)
        return x  

class DATA_LOADER(object):
    def __init__(self, opt):

        opt.use_sentence = False # use text-description embedding as auxiliary information
        self.use_sentence = (opt.use_sentence and opt.dataset == 'CUB')
        self.read_matdataset(opt)

    def read_matdataset(self, opt):
        matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.image_embedding + ".mat")
        self.feature = matcontent['features'].T
        self.label = matcontent['labels'].astype(int).squeeze() - 1
        matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.class_embedding + "_splits.mat")
        trainval_loc = matcontent['trainval_loc'].squeeze() - 1
        test_seen_loc = matcontent['test_seen_loc'].squeeze() - 1
        test_unseen_loc = matcontent['test_unseen_loc'].squeeze() - 1    

        if self.use_sentence:
            meta = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + "data.mat")
            attribute_np = meta['attribute'] 
        else:
            attribute_np = matcontent['att'].T
        self.get_mean()
        self.attribute = torch.from_numpy(attribute_np).float()



        trainval =self.feature[trainval_loc]
        test_unseen =self.feature[test_unseen_loc]
        test_seen= self.feature[test_seen_loc]


        if opt.preprocessing:
            scaler = preprocessing.MinMaxScaler()
            _train_feature = scaler.fit_transform(trainval)
            _test_seen_feature = scaler.transform(test_seen)
            _test_unseen_feature = scaler.transform(test_unseen)
            self.train_feature = torch.from_numpy(_train_feature).float()
            mx = self.train_feature.max()
            self.train_feature.mul_(1/mx)
            self.train_label = torch.from_numpy(self.label[trainval_loc]).long() 
            self.test_unseen_feature = torch.from_numpy(_test_unseen_feature).float()
            self.test_unseen_feature.mul_(1/mx)
            self.test_unseen_label = torch.from_numpy(self.label[test_unseen_loc]).long() 
            self.test_seen_feature = torch.from_numpy(_test_seen_feature).float() 
            self.test_seen_feature.mul_(1/mx)
            self.test_seen_label = torch.from_numpy(self.label[test_seen_loc]).long()

        else:
            self.train_feature = torch.from_numpy(trainval).float()
            self.train_label = torch.from_numpy(self.label[trainval_loc]).long() 
            self.test_unseen_feature = torch.from_numpy(test_unseen).float()
            self.test_unseen_label = torch.from_numpy(self.label[test_unseen_loc]).long() 
            self.test_seen_feature = torch.from_numpy(test_seen).float() 
            self.test_seen_label = torch.from_numpy(self.label[test_seen_loc]).long()

        self.test_feature = torch.cat([self.test_seen_feature,self.test_unseen_feature],dim=0)
        self.test_label = torch.cat([self.test_seen_label,self.test_unseen_label])
        
        self.test_classes_np = np.arange(len(self.attribute))
        self.seenclasses = torch.from_numpy(np.unique(self.train_label.numpy()))
        self.unseenclasses = torch.from_numpy(np.unique(self.test_unseen_label.numpy()))

        self.ntrain = self.train_feature.size()[0]
        self.ntest_seen = self.test_seen_feature.size()[0]
        self.ntest_unseen = self.test_unseen_feature.size()[0]
        self.ntest = self.ntest_seen + self.ntest_unseen
        self.ntrain_class = self.seenclasses.size(0)
        self.ntest_class = self.unseenclasses.size(0)
        self.train_class = self.seenclasses.clone()
        self.nclass = self.ntest_class + self.ntrain_class
        self.allclasses = torch.arange(0, self.ntrain_class + self.ntest_class).long()
        
        self.train_mapped_label = map_label(self.train_label, self.seenclasses) 
        self.test_seen_mapped_label = map_label(self.test_seen_label,self.seenclasses)
        self.test_unseen_mapped_label = map_label(self.test_unseen_label, self.unseenclasses)
        unseen_class_counts = np.unique(self.test_unseen_label.numpy(),return_counts=True)[1]
        self.real_class_prior = unseen_class_counts/unseen_class_counts.sum()
        self.seen_classes_dict = []
        for seen_class in range(len(self.seenclasses)):
            self.seen_classes_dict.append(np.where(self.train_mapped_label.numpy()==seen_class))

        self.unseen_classes_dict = []
        self.unseen_prior = []
        for unseen_class in range(len(self.unseenclasses)):
            unseen_idx = np.where(self.test_unseen_mapped_label.numpy()==unseen_class)[0]
            self.unseen_classes_dict.append(unseen_idx)
            self.unseen_prior.append(len(unseen_idx))
        self.unseen_prior = np.array(self.unseen_prior)/np.array(self.unseen_prior).sum()
        logger.info(f"unseen_prior:{self.unseen_prior}")
        
        self.seen_att = self.attribute[self.seenclasses]
        self.unseen_att = self.attribute[self.unseenclasses]
        self.novel_att = self.unseen_att.mean(0) 

        if opt.L2_norm:
            assert opt.preprocessing is False 
            logger.info(f"unseen_feature:{torch.norm(self.test_unseen_feature,p=2,dim=1).mean()}")
            logger.info(f"att:{torch.norm(self.attribute ,p=2,dim=1).mean()}")
            self.norm_feature(radius=opt.radius)
            self.attribute = norm(1)(self.attribute)   # attribute normalization
            logger.info(f"unseen_feature_normed:{torch.norm(self.test_unseen_feature,p=2,dim=1).mean()}")
            logger.info(f"att_normed:{torch.norm(self.attribute ,p=2,dim=1).mean()}")

    def get_mean(self):
        self.mean = []
        for i in range(len(np.unique(self.label))):
            self.mean.append(self.feature[np.where(self.label == i)].mean(0))
        self.mean = np.stack(self.mean)

    def save_feature(self,path):
        feature_dict = {'train_seen':self.train_feature,
                        'test_seen':self.test_seen_feature,
                        'test_unseen':self.test_unseen_feature,
                        }
        torch.save(feature_dict,path)
        print('new data save at {}'.format(path))

    def cal_Unseen_meanVar(self):
        self.unseen_meanStatics=[]
        self.unseen_varStatics=[]
        for i, idx in enumerate(self.unseen_classes_dict):
            self.unseen_meanStatics.append(torch.mean(self.test_unseen_feature[idx],dim=0))
            self.unseen_varStatics.append(torch.var(self.test_unseen_feature[idx],dim=0))
        self.unseen_meanStatics=torch.stack(self.unseen_meanStatics,dim=0)
        self.unseen_varStatics=torch.stack(self.unseen_varStatics,dim=0)

    def next_unseen_batch(self, seen_batch,perb = False,unknown_prior=False,unseen_prior = None, ):
        idx = torch.randperm(self.ntest_unseen)[0:seen_batch]
        batch_feature = self.test_unseen_feature[idx]
        if not unknown_prior:
            idx2 = torch.randperm(self.ntest_unseen)[0:seen_batch]
            batch_ran_label = self.test_unseen_label[idx2]
            batch_ran_mapped_label = self.test_unseen_mapped_label[idx2]
            batch_ran_att = self.attribute[batch_ran_label]
        else:
            if unseen_prior is not None:
                # Sampling from given class freequency.
                unseen_prior = unseen_prior/unseen_prior.sum(0)
                batch_ran_label = np.random.choice(self.unseenclasses,seen_batch,p = unseen_prior)
                self.batch_ran_mapped_label = map_label(torch.from_numpy(batch_ran_label) , self.unseenclasses)
            else:
                # Uniform sampling
                batch_ran_label = np.random.choice(self.unseenclasses,seen_batch)
            batch_ran_att = self.attribute[batch_ran_label]

        if perb:
            batch_ran_att = perb_att(batch_ran_att)
        
        return batch_feature, batch_ran_att

    def next_gzsl_unseen_batch(self, seen_batch,unknown_prior=False,unseen_prior = None):
        idx = torch.randperm(self.ntest)[0:seen_batch]
        batch_feature = self.test_feature[idx]
        idx2 = torch.randperm(self.ntest)[0:seen_batch]
        batch_ran_label = self.test_label[idx2]
        batch_ran_att = self.attribute[batch_ran_label]
        return batch_feature, batch_ran_att   
    
    def next_seen_batch(self, seen_batch, perb = False ,return_mapped_label = False,):
        idx = torch.randperm(self.ntrain)[0:seen_batch]
        batch_feature = self.train_feature[idx]
        batch_label = self.train_label[idx]
        batch_att = self.attribute[batch_label]
        if perb:
            batch_att = perb_att(batch_att)
        if return_mapped_label:
            batch_mapped_label = self.train_mapped_label[idx]
            return batch_feature, batch_att,batch_mapped_label
        return batch_feature, batch_att

    def norm_feature(self,radius=1):
        self.train_feature = radius * self.train_feature/torch.norm(self.train_feature,p=2,dim=1,keepdim=True)
        self.test_unseen_feature = radius * self.test_unseen_feature/torch.norm(self.test_unseen_feature,p=2,dim=1,keepdim=True)
        self.test_seen_feature = radius * self.test_seen_feature/torch.norm(self.test_seen_feature,p=2,dim=1,keepdim=True)
        self.cal_Unseen_meanVar()
        



