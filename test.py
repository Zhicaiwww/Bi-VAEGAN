import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np 
import matplotlib.cm as cm
import torch
import copy
from datasets.ZSLDataset import DATA_LOADER
from configs import opt
from networks.VAEGANV1_model import Decoder,AttR
import classifiers.classifier_ZSL as classifier
import random
# from visual import tsne_visual, tsne_visual_att
torch.cuda.set_device('cuda:5')
def visual_mapped_att(netR,mode = 2,path='de'):
    if mode ==1:
        savepath = path + '_u+s.pdf' 
        syn_TrueUnseen_att = netR(torch.from_numpy(unseen_true_test_feature).cuda())
        syn_TrueSeen_att = netR(torch.from_numpy(seen_true_test_feature).cuda())
        all_att = np.array(data.attribute)
        dataset_x = torch.cat([syn_TrueUnseen_att,syn_TrueSeen_att],dim=0)
        dataset_x = np.concatenate([np.array(dataset_x.detach().cpu()),all_att],axis=0)
        label = np.concatenate([unseen_true_test_label,seen_true_test_label,np.arange(len(all_att))],axis=0)
        mark_1 = np.ones_like(unseen_true_test_label)
        mark_2 = np.zeros_like(seen_true_test_label)
        mark_3 = np.ones_like(np.arange(len(all_att)))
        mark_3.fill(2)

        mark = np.concatenate([mark_1,mark_2,mark_3],axis=0)
        print(dataset_x.shape,label.shape,mark.shape)
        tsne_visual(dataset_x=dataset_x,label = label, mark=mark, path = savepath)
    elif mode ==2:
        savepath = path + '_u.pdf' 
        syn_TrueUnseen_att = netR(torch.from_numpy(unseen_true_test_feature).cuda())
        all_att = np.array(data.unseen_att)
        
        dataset_x = syn_TrueUnseen_att
        dataset_x = np.concatenate([np.array(dataset_x.detach().cpu()),all_att],axis=0)
        label = np.concatenate([unseen_true_test_label,np.array(data.unseenclasses)],axis=0)
        mark_1 = np.ones_like(unseen_true_test_label)
        mark_3 = np.ones_like(np.arange(len(all_att)))
        mark_3.fill(2)

        mark = np.concatenate([mark_1,mark_3],axis=0)
        print(dataset_x.shape,label.shape,mark.shape)
        tsne_visual(dataset_x=dataset_x,label = label, mark=mark, path = savepath)


def generate_syn_feature(netG, classes, attribute, num):
    nclass = classes.size(0)
    syn_feature = torch.FloatTensor(nclass*num, opt.resSize)
    syn_label = torch.LongTensor(nclass*num) 
    syn_att = torch.FloatTensor(num, opt.attSize)
    syn_noise = torch.FloatTensor(num, opt.attSize)
    if opt.cuda:
        syn_att = syn_att.cuda()
        syn_noise = syn_noise.cuda()
        
    with torch.no_grad():
        for i in range(nclass):
            iclass = classes[i]
            iclass_att = attribute[iclass]
            syn_att.copy_(iclass_att.repeat(num, 1))
            syn_noise.normal_(0, sigma)
            output = netG(syn_noise, c= syn_att)
            syn_feature.narrow(0, i*num, num).copy_(output.data.cpu())
            syn_label.narrow(0, i*num, num).fill_(iclass)

    return syn_feature, syn_label


if __name__ == '__main__':
    opt.dataset = 'AwA2'
    tag = 'E'
    opt.resSize = 2048
    if opt.dataset == 'AWA1':
        num_classes = 50
        opt.attSize = 85 
        path_att = r'/home/zhicai/bivaegan/TZSL/out/AWA1/Q1_2022-11-06_fVGan_Perb_RW10.0_Beta1.0_RD10.0_RG1.0_iter3_TransD100_TransG10_MW1.0_r2_Ar1_Seed9122_W1/acc_0.9394033551216125.pth'
        opt.manualSeed = 9182
    if opt.dataset == 'AwA2':
        num_classes = 50
        opt.attSize = 85
        path_att = r'/home/zhicai/bivaegan/TZSL/out/AwA2/Q_T_2022-09-16_fVGan_Perb_RW1.0_RD10.0_RG1.0_iter3_TransD100_TransG10_MW1.0_r3_Ar1_Seed9182_W1/acc_0.9570616483688354.pth' 
        # path_att = r'/home/zhicai/bivaegan/TZSL/out/AwA2/Q3_2022-09-27_fVGan_Perb_RW100.0_RD10.0_RG1.0_iter3_TransD100_TransG10_MW1.0_r1_Ar1_Seed9152_W1/acc_0.9469022750854492.pth'
        path_att = r'/home/zhicai/bivaegan/TZSL/out/AwA2/Q_T_2022-09-14_fVGan_Perb_RW10.0_RD10.0_RG1.0_iter3_TransD100_TransG10_MW1.0_r1_Ar1_Seed9182_W1/acc_0.949701189994812.pth'
        opt.manualSeed = 9182
    if opt.dataset == 'CUB':
        num_classes = 200
        opt.attSize = 1024
        opt.manualSeed = 3483
        path_att= r'/home/zhicai/bivaegan/TZSL/out/CUB/ZSL_2022-09-29_fVGan_RW10.0_RD10.0_RG0.1_iter3_eN_TransD1000_TransG10_MW1.0_r1_Ar1_Seed4115_W1/acc_0.7681369781494141.pth'
        path_att = r'/home/zhicai/bivaegan/TZSL/out/CUB/ZSL_sen_2022-09-29_fVGan_RW10.0_RD10.0_RG0.1_iter3_eN_TransD1000_TransG10_MW1.0_r3_Ar1_Seed4115_W1/acc_0.8266139030456543.pth'
    if opt.dataset == 'FLO':
        num_classes = 102
        opt.attSize = 1024
    if opt.dataset == 'SUN':
        num_classes = 717
        opt.manualSeed = 4115
        opt.attSize = 102
        path_att = r'/home/zhicai/bivaegan/TZSL/out/SUN/Q3_2022-09-19_fVGan_Perb_RW10.0_RD1.0_RG0.01_iter3_eN_TransD100_TransG1_MW1.0_r1_Ar1_Seed4115_W1/acc_0.7506945133209229.pth'
    opt.preprocessing = False
    opt.L2_norm=True
    opt.dataroot='/home/zhicai/data'
    opt.image_embedding='res101'
    opt.radius=1
    mode=2 
    opt.cuda = True
    opt.classifier_lr=0.002
    opt.classifier_glr = 0.01
    sigma = 1.2
    syn_num =300
    syn_true_num = 0  
    cm_epoch = 0
    no_R = False
    # path_att = r'/home/zhicai/bivaegan/TZSL/out/AwA2/Q3_2022-09-27_fVGan_Perb_RW10.0_RD10.0_RG1.0_iter3_MW1.0_r1_Ar1_Seed9152_W1/acc_0.6758228540420532.pth'

    # 
    # 
    # path_att = r'/home/zhicai/bivaegan/TZSL/out/AWA1/Q_2022-09-21_fVGan_Perb_RW1.0_RD10.0_RG1.0_iter3_TransD100_TransG10_MW1.0_r1_Ar1_Seed9182_W1/acc_0.9118145108222961.pth'
    # 
    # path_att = r'/home/zhicai/bivaegan/TZSL/out/CUB/ZSL_2022-09-27_fVGan_TuneReA1_RW10.0_RD10.0_RG0.1_iter3_eN_TransD1000_TransG10_MW1.0_r3_Ar1_Seed4115_W1/acc_0.7782032489776611.pth'
    data = DATA_LOADER(opt)
    # initialize generator and discriminator
    netG_att = Decoder(opt).cuda()
    netR = AttR(opt).cuda()
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)


    # path_onto = 'out/ckpts/debug_ATT_DEC.pth'
    # state_dict = torch.load(path_onto)['netG_state_dict']
    # netG_onto.load_state_dict(state_dict)

    ckpt = torch.load(path_att)
    state_dict = ckpt['netG_state_dict']
    netG_att.load_state_dict(state_dict)

    if no_R:
        netR = None
    else:
        netR_c = ckpt['netR_state_dict']
        netR.load_state_dict(netR_c)
        
    unseen_classes = data.unseenclasses
    seen_classes = data.seenclasses

    unseen_true_test_feature = np.array(data.test_unseen_feature)
    unseen_true_test_label = np.array(data.test_unseen_label)
    seen_true_test_feature = np.array(data.test_seen_feature)
    seen_true_test_label = np.array(data.test_seen_label)
    seen_true_train_label = np.array(data.train_label)
    seen_true_train_feature = np.array(data.train_feature)


    
    # if opt.dataset == 'CUB':
    #     path2 = '/home/zhicai/bivaegan/TZSL/datasets/CUB/ZSL_2022-09-27_fVGan_TuneReA1_RW10.0_RD10.0_RG0.1_iter3_eN_TransD1000_TransG10_MW1.0_r3_Ar1_Seed4115_W1_Epo12_acc0.7245_r3.pth'
    #     data.train_feature = torch.load(path2)['train_seen']
    #     data.test_unseen_feature = torch.load(path2)['test_unseen']
    #     data.test_seen_feature = torch.load(path2)['test_seen']
        
    att = data.attribute 
    # onto = data.onto_embedding
    data_att = att
    syn_feature, syn_label = generate_syn_feature(netG_att,data.unseenclasses,data_att, syn_num)

    if syn_true_num:
        syn_true_featrue, syn_true_label = generate_syn_feature(netG_att,data.seenclasses,data_att,num = syn_true_num) 
        train_X = torch.cat((data.train_feature,syn_true_featrue, syn_feature), 0)
        train_Y = torch.cat((data.train_label,syn_true_label, syn_label), 0)
    else:
        train_X = torch.cat((data.train_feature, syn_feature), 0)
        train_Y = torch.cat((data.train_label, syn_label), 0)

        # tsne_visual_att(netR)
    print(np.linalg.norm(train_X,axis=1))
    # gzsl_cls = classifier.CLASSIFIER(train_X, train_Y, data, data.nclass, opt.cuda, opt.classifier_glr, 0.5, \
                    # 60, syn_num, netR=netR,dec_size=opt.attSize,generalized=True)
    from datasets import ZSLDataset as util
    zsl_cls = classifier.CLASSIFIER(syn_feature, util.map_label(syn_label, data.unseenclasses), \
                    data, data.unseenclasses.size(0), opt.cuda,  opt.classifier_lr, 0.5, 50, syn_num,netR=netR,\
                        dec_size=opt.attSize, generalized=False,feature_type='a')
    print(zsl_cls.acc)
    # print(zsl_cls.per_acc)
    
        

