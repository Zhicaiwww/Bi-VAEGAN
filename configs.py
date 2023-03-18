#author: akshitac8
import argparse
import os
import datetime
import logging
from helper import get_logger
class OPT ():
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--dataset', default='CUB', help='FLO')
        parser.add_argument('--dataroot', default='/home/zhicai/data', help='path to dataset')
        parser.add_argument('--image_embedding', default='res101')
        parser.add_argument('--class_embedding', default='att')
        parser.add_argument('--syn_num', type=int, default=100, help='number features to generate per class')
        parser.add_argument('--syn_num2', type=int, default=100, help='number features to generate per class')
        parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
        parser.add_argument('--resSize', type=int, default=2048, help='size of visual features')
        parser.add_argument('--attSize', type=int, default=1024, help='size of semantic features')
        parser.add_argument('--nepoch', type=int, default=2000, help='number of epochs to train for')
        parser.add_argument('--critic_iter', type=int, default=5, help='critic iteration, following WGAN-GP')
        parser.add_argument('--lambda1', type=float, default=10, help='gradient penalty regularizer, following WGAN-GP')
        parser.add_argument('--lambda2', type=float, default=10, help='gradient penalty regularizer, following WGAN-GP')
        parser.add_argument('--lr', type=float, default=0.001, help='learning rate to train GANs ')
        parser.add_argument('--classifier_lr', type=float, default=0.001, help='learning rate to train softmax classifier')
        parser.add_argument('--cuda', action='store_true', default=True, help='enables cuda')
        parser.add_argument('--encoded_noise', action='store_true', default=False, help='enables validation mode')
        parser.add_argument('--manualSeed', type=int, help='manual seed')
        parser.add_argument('--nclass_all', type=int, default=200, help='number of all classes')
        parser.add_argument('--gammaD', type=int, default=1000, help='weight on the W-GAN loss')
        parser.add_argument('--gammaG', type=int, default=1000, help='weight on the W-GAN loss')
        parser.add_argument('--gammaG_un', type=int, default=1000, help='weight on the W-GAN loss')
        parser.add_argument('--gammaD_un', type=int, default=1000, help='weight on the W-GAN loss')
        parser.add_argument('--preprocessing', action='store_true', default=False, help='enbale MinMaxScaler on visual features')
        parser.add_argument('--gzsl', action='store_true', default=False)
        ###
        parser.add_argument('--transductive', action='store_true', default=False)
        parser.add_argument('--RCritic', action='store_true', default=False, help = 'enable use RCritic ')
        parser.add_argument('--beta', type=float, default=1.0, help='beta for objective L_R')
        parser.add_argument('--L2_norm', action='store_true', default=False, help='enbale L_2 nomarlization on visual features')
        parser.add_argument('--radius', type=int, default=1, help='radius of L_2 feature nomalization')
        parser.add_argument('--att_criterian', type=str, default='W1')
        parser.add_argument('--gammaD_att', type=float, default=10.0, help='weight on the W-GAN loss')
        parser.add_argument('--gammaG_att', type=float, default=0.1, help='weight on the W-GAN loss')
        parser.add_argument('--perb', action='store_true', default=False)

        parser.add_argument('--pretune_feature', action='store_true', default=False, help='enable pre-tune visual features')
        parser.add_argument('--tune_epoch', type=int, default=15, help = 'pretune epochs')    


        parser.add_argument('--unknown_classDistribution', action='store_true', default=False, help='training in the unknown class distribution for the unseen classes')
        parser.add_argument('--no_R', action='store_true', default=False, help='no use regressor module')
        parser.add_argument('--soft', action='store_true', default=False)
        parser.add_argument('--ind_epoch', type=int, default=3, help='inductive epoch')
        parser.add_argument('--prior_estimation', type=str, default="CPE", help='CPE or BBSE or classifier')

        opt, _ = parser.parse_known_args()
        self.opt, self.log_dir, self.logger, self.training_logger = self.set_opt(opt)

    def return_opt(self):
        return self.opt, self.log_dir, self.logger, self.training_logger

    def set_opt(self,opt):
    
        opt.tag = ''
        opt.R = ~ opt.no_R
        opt.lambda2 = opt.lambda1
        opt.tag += f'{datetime.date.today()}_seed{opt.manualSeed}'

        if opt.unknown_classDistribution:
            opt.tag+=f'_noPrior+{opt.prior_estimation}'
        if opt.pretune_feature:
            opt.tag+=f'_pretuned'
        if opt.R:
            if opt.RCritic:
                opt.tag +=f'_RD{opt.gammaD_att}_RG{opt.gammaG_att}'
        else:
            opt.tag+="_noR"

        if opt.transductive:
            opt.tag += f'_TransD{opt.gammaD_un}_TransG{opt.gammaG_un}'
            opt.tag +=f'_RW{opt.beta}'
        if opt.L2_norm:
            opt.tag += f'_r{opt.radius}'
        log_dir = os.path.join('out', f'{opt.dataset}',f'{opt.tag}')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        logging.basicConfig(level=logging.DEBUG)
        logger = logging.getLogger('train')
        console = logging.StreamHandler()
        console.setLevel("INFO")
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - \n%(message)s")
        console.setFormatter(formatter)
        handler = logging.FileHandler(f'{log_dir}/log.txt')
        handler.setLevel('INFO')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.info(f'save at {log_dir}')
    
        training_logger = get_logger(log_dir)
        return opt,log_dir,logger, training_logger