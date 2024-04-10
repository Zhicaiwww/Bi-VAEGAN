import tqdm
import os
import random
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import sys
sys.path.append(os.getcwd())
from datasets.ZSLDataset import DATA_LOADER, map_label
import classifiers.classifier_ZSL as classifier
from networks import VAEGANV1_model as model
import numpy as np
from configs import OPT
from networks.pretune import pretune
from networks.label_shift import ls
from networks.utils import generate_syn_feature, loss_fn, loss_fn_2, calc_gradient_penalty

opt,log_dir,logger, training_logger = OPT().return_opt()

opt.tr_sigma = 1.0

if opt.gzsl==True:
    assert opt.unknown_classDistribution is False

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
logger.info(f'{opt}')
logger.info('Random Seed=%d\n' % (opt.manualSeed))
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)
cudnn.benchmark = True

# load data
data = DATA_LOADER(opt)

if opt.pretune_feature:
    pretune(opt,data,save=True)

netG = model.Decoder(opt).cuda()
netCritic = model.MLP_CRITIC(opt).cuda()
netR = model.AttR(opt).cuda()
netE = model.Encoder(opt).cuda()
netCritic_un = model.MLP_CRITIC_un(opt).cuda() 
netRCritic = model.netRCritic(opt).cuda()
    
logger.info(netCritic_un)
logger.info(netE)
logger.info(netR)
logger.info(netCritic)
logger.info(netG)

input_res = torch.FloatTensor(opt.batch_size, opt.resSize).cuda()
input_res_novel = torch.FloatTensor(opt.batch_size, opt.resSize).cuda()
input_att = torch.FloatTensor(opt.batch_size, opt.attSize).cuda()
input_att_novel = torch.FloatTensor(opt.batch_size, opt.attSize).cuda()
input_novel_mlabel = torch.LongTensor(opt.batch_size).cuda()
noise_att = torch.FloatTensor(opt.batch_size, opt.attSize).cuda()
sample_att = torch.FloatTensor(opt.batch_size, opt.attSize).cuda()
one = torch.tensor(1, dtype=torch.float).cuda()
mone = one * -1


def sample_unseen(perb=False, unknown_prior=False,unseen_prior=None):
    batch_data,batch_att= data.next_unseen_batch(opt.batch_size,unknown_prior = unknown_prior,unseen_prior=unseen_prior,perb=perb)
    input_res_novel.copy_(batch_data)
    input_att_novel.copy_(batch_att)

def sample(perb = False):
    batch_feature,batch_att= data.next_seen_batch(opt.batch_size,perb=perb)
    input_res.copy_(batch_feature)
    input_att.copy_(batch_att)

def zero_grad(p):
    if p.grad is not None:
        p.grad.detach_()
        p.grad.zero_()
        
def freezenet(net):
    for p in net.parameters(): 
        p.requires_grad = False 

def trainnet(net):
    for p in net.parameters(): 
        p.requires_grad = True 

# setup optimizer
optimizerCritic = optim.AdamW(netCritic.parameters(), lr=opt.lr, betas=(0.5, 0.999))
optimizerCritic_un = optim.AdamW(netCritic_un.parameters(),lr=opt.lr,betas = (0.5, 0.999))
optimizerG = optim.AdamW(netG.parameters(), lr=opt.lr, betas=(0.5, 0.999))
optimizerE_att = optim.AdamW(netE.parameters(), lr=opt.lr, betas=(0.5, 0.999))
optimizerRCritic = optim.AdamW(netRCritic.parameters(), lr=opt.lr, betas=(0.5, 0.999))
optimizerR = optim.AdamW(netR.parameters(), lr=opt.lr, betas=(0.5, 0.999))

best_gzsl_acc=0
best_zsl_acc = 0
best_acc_seen=0
best_acc_unseen=0

pre_path = None
class_prior = None

if opt.R:
    feature_type = 'vha'
else: 
    feature_type = 'v'
    netR = None
    
for epoch in range(opt.nepoch):
    
    if opt.transductive:
        if epoch < opt.ind_epoch and opt.unknown_classDistribution:
            print('----'*8,'Inductive Training','----'*8)
            use_transductive_training = False
            class_prior = None
        else:
            print('----'*8,'Transductive Training','----'*8)
            use_transductive_training = True
    
    for i, batch_idx in tqdm.tqdm(enumerate(range(0, data.ntrain, opt.batch_size)),desc = 'Trainging Epoch {}'.format(epoch)):
        # Step 1 -----------------------------------------------------------------------------
        ### Train attribute regressor 


        if i % 5 == 0 and opt.R:
            
            if opt.RCritic and use_transductive_training:
                ### Train attribute critic transductively

                trainnet(netRCritic)
                freezenet(netR)
                # Dafault set 3 
                for j in range(5):
                    netRCritic.zero_grad()
                    # Encode seen attribute in RCritic
                    sample(perb = opt.perb)
                    CriticR_real_seen = opt.gammaD_att * netRCritic(input_att).mean()
                    CriticR_real_seen.backward(mone)
                    input_att_fakeSeen = netR(input_res).detach()
                    CriticR_fake_seen = opt.gammaD_att * netRCritic(input_att_fakeSeen).mean()
                    CriticR_fake_seen.backward(one)
                    # Train unseen attribute RCritic
                    sample_unseen(perb=opt.perb, unknown_prior=opt.unknown_classDistribution,unseen_prior=class_prior)
                    CriticR_real_unseen = opt.gammaD_att * netRCritic(input_att_novel).mean()
                    CriticR_real_unseen.backward(mone)
                    input_att_fakeUnSeen = netR(input_res_novel).detach()
                    CriticR_fake_unseen = opt.gammaD_att * netRCritic(input_att_fakeUnSeen).mean()
                    CriticR_fake_unseen.backward(one)

                    # Gradient penalty
                    input_att_all = torch.cat([input_att,input_att_novel],dim=0)
                    fake_att_all = torch.cat([input_att_fakeSeen,input_att_fakeUnSeen],dim=0)
                    gradient_penalty_att = opt.gammaD_att * calc_gradient_penalty(opt,netRCritic, input_att_all, fake_att_all.data,lambda1 = 0.1)
                    gradient_penalty_att.backward()

                    Wasserstein_R_attUnseen = CriticR_real_unseen - CriticR_fake_unseen
                    optimizerRCritic.step()
                    training_logger.update_meters(['criticR/GP_att','criticR/WD_unseen'],\
                    [gradient_penalty_att.item(),Wasserstein_R_attUnseen.item()],input_res.size(0))
                freezenet(netRCritic)
                
            trainnet(netR)
            freezenet(netG)
            
            for _ in range(5):
                ### Train attribute critic supervisedly
                sample()
                netR.zero_grad()
                R_loss, mapped_seen_att = netR(input_res,input_att)
                training_logger.update_meters(['R/loss'], [R_loss.item()],input_res.size(0))

                if opt.RCritic and use_transductive_training:
                    ### Train attribute critic transductively
                    sample_unseen(unknown_prior=opt.unknown_classDistribution,unseen_prior=class_prior)
                    mapped_unseen_att = netR(input_res_novel)

                    G_loss_R =  netRCritic(mapped_seen_att).mean() + netRCritic(mapped_unseen_att).mean()
                    R_loss+= -opt.gammaG_att *G_loss_R

                    training_logger.update_meters(['R/G_loss_R'],[opt.gammaG_att * G_loss_R.item()],input_res.size(0))
                R_loss =  R_loss
                R_loss.backward()
                optimizerR.step()

        trainnet(netG)
        trainnet(netCritic)
        trainnet(netCritic_un)
        if opt.R:
            freezenet(netR)

        gp_sum = 0 #lAMBDA VARIABLE
        gp_sum2 = 0 

        # Step 2 -----------------------------------------------------------------------------
        for _ in range(opt.critic_iter):
            sample()

            ### Train conditional Critic of the seen classes
            netCritic.zero_grad()
            if opt.encoded_noise:        
                means, log_var = netE(input_res, input_att)
                std = torch.exp(0.5 * log_var)
                eps = torch.randn([opt.batch_size, opt.attSize]).cpu()
                eps = Variable(eps.cuda())
                z = eps * std + means 
            else:
                noise_att.normal_(0, opt.tr_sigma)
                z = Variable(noise_att)
            fake = netG(z,input_att)
            criticD_real = netCritic(input_res, input_att)
            criticD_real = opt.gammaD* criticD_real.mean()
            criticD_real.backward(mone)
            # train with fake seen feature
            criticD_fake = netCritic(fake.detach(), input_att)
            criticD_fake = opt.gammaD* criticD_fake.mean()
            criticD_fake.backward(one)
            # gradient penalty
            gradient_penalty = opt.gammaD* calc_gradient_penalty(opt, netCritic, input_res, fake.data, input_att = input_att,lambda1 =opt.lambda1)
            gradient_penalty.backward()
            gp_sum += gradient_penalty.data /1.0

            Wasserstein_D = criticD_real - criticD_fake
            optimizerCritic.step()
            training_logger.update_meters(['criticD/WGAN','criticD/GP_att'],[Wasserstein_D.item(),gradient_penalty.item()],input_res.size(0))
            ### train unconditional Critic
            if use_transductive_training:
                netCritic_un.zero_grad()
                sample_unseen(unknown_prior=opt.unknown_classDistribution,unseen_prior=class_prior)
                criticD_un_real = netCritic_un(input_res_novel).mean()
                criticD_un_real = opt.gammaD_un*criticD_un_real
                criticD_un_real.backward(mone)
                # train with fakeG
                noise_att.normal_(0, opt.tr_sigma)
                fake_novel = netG(noise_att,input_att_novel)
                criticD_un_fake = netCritic_un(fake_novel)
                
                criticD_un_fake = opt.gammaD_un*criticD_un_fake.mean()
                criticD_un_fake.backward(one)
                # gradient penalty
                gradient_un_penalty =  opt.gammaD_un*calc_gradient_penalty(opt, netCritic_un, input_res_novel, fake_novel.data,lambda1 =opt.lambda2)
                gradient_un_penalty.backward()
                gp_sum2 += gradient_un_penalty.data 
                Wasserstein_D_un = criticD_un_real - criticD_un_fake
                optimizerCritic_un.step()
                training_logger.update_meters(['criticD2/WGAN','criticD2/GP_att',],[Wasserstein_D_un.item(),gradient_un_penalty.item(),],input_res.size(0))
        
        gp_sum /= (opt.gammaD*opt.lambda1*opt.critic_iter)       
        if (gp_sum > 1.05).sum() > 0:
            opt.lambda1 *= 1.1
        elif (gp_sum < 1.001).sum() > 0:
            opt.lambda1 /= 1.1
        training_logger.update_meters(['criticD/lambda1',],[opt.lambda1],input_res.size(0))

        if use_transductive_training:
            gp_sum2 /= (opt.gammaD_un*opt.lambda2*opt.critic_iter)
            if (gp_sum2 > 1.05).sum() > 0:
                opt.lambda2 *= 1.1
            elif (gp_sum2 < 1.001).sum() > 0:
                opt.lambda2 /= 1.1 
            training_logger.update_meters(['criticD2/lambda2',],[opt.lambda2],input_res.size(0))
            freezenet(netCritic_un)

        freezenet(netCritic)

        # Step 3 -----------------------------------------------------------------------------
        # Train generator
        netG.zero_grad()
        netE.zero_grad()
        mean_1, log_var_1 = netE(input_res, input_att)
        std_1 = torch.exp(0.5 * log_var_1)
        latent_1 = mean_1.size(1)
        eps_1 = torch.randn([opt.batch_size,latent_1 ]).cuda()
        z_1 = eps_1 * std_1 + mean_1
        
        # VAE reconstruction loss
        if opt.L2_norm:
            recon_x= netG(z_1,input_att)
            recon_x_Notnormed = netG.get_out()
            recon_x_Notnormed = torch.norm(recon_x_Notnormed,dim = -1).sum().item()/ input_res.size(0)
            training_logger.update_meters(['Visualization/seen_norm'],[recon_x_Notnormed],1)
            vae_loss = loss_fn_2(opt, recon_x, input_res, mean_1, log_var_1)
        else: 
            recon_x = netG(z_1,input_att)
            vae_loss = loss_fn(opt, recon_x, input_res, mean_1, log_var_1)

        # Align conditional seen generation to intra-class distribution.
        if opt.encoded_noise :
            criticG_recon_x_loss =  netCritic(recon_x, input_att).mean()
            fake_v = recon_x
            criticG_fake_loss = criticG_recon_x_loss
        else:
            noise_att.normal_(0, opt.tr_sigma)
            fake_v = netG(noise_att,input_att)
            criticG_fake_loss =  netCritic(fake_v, input_att).mean()

        loss = vae_loss - opt.gammaG * criticG_fake_loss   


        training_logger.update_meters(['G/fakeG_loss','G/vae_loss'],[- opt.gammaG *criticG_fake_loss.item(),vae_loss.item()],input_res.size(0))         
        if use_transductive_training and opt.R :
            # ReMap conditional unseen generation to its conditioned attribute  .
            noise_att.normal_(0, opt.tr_sigma)
            fake_novel = netG(noise_att,input_att_novel.detach())
            fake_novel_D_loss = netCritic_un(fake_novel)
            fake_novel_D_loss = fake_novel_D_loss.mean()
            loss += -opt.gammaG_un * fake_novel_D_loss
            R_loss_unseen_fake, mapped_Gunseen_att = netR(fake_novel,input_att_novel)
            loss += opt.beta * R_loss_unseen_fake 

        loss.backward()
        optimizerG.step()
        optimizerE_att.step()

    training_logger.flush_meters(epoch)

    # Evaluate the model, set G to evaluation mode
    netG.eval()
    if opt.R:
        netR.eval()

    syn_feature, syn_label ,out_notNorm = generate_syn_feature(opt, netG,data.unseenclasses, data.attribute, opt.syn_num, return_norm = True)
    out_notNorm = torch.norm(out_notNorm, dim = -1).sum().item()/ out_notNorm.size(0)
    training_logger.update_meters(['Visualization/unseen_norm'],[out_notNorm],1)

    if opt.gzsl:
        # Concatenate real seen features with synthesized unseen features
        train_X = torch.cat((data.train_feature, syn_feature), 0)
        train_Y = torch.cat((data.train_label, syn_label), 0)

        nclass = opt.nclass_all
        # Train GZSL classifier
        gzsl_cls = classifier.CLASSIFIER(train_X, train_Y, data, nclass, opt.cuda, 0.001, 0.5, \
                20, opt.syn_num, netR=netR,dec_size=opt.attSize,generalized=True)
        if best_gzsl_acc < gzsl_cls.H:
            best_acc_seen, best_acc_unseen, best_gzsl_acc = gzsl_cls.acc_seen, gzsl_cls.acc_unseen, gzsl_cls.H

        logger.info('GZSL: seen=%.4f, unseen=%.4f, h=%.4f' % (gzsl_cls.acc_seen, gzsl_cls.acc_unseen, gzsl_cls.H))

    # else:
    zsl_cls = classifier.CLASSIFIER(syn_feature, map_label(syn_label, data.unseenclasses), \
                data, data.unseenclasses.size(0), opt.cuda, opt.classifier_lr, 0.5, 25, opt.syn_num,netR=netR,\
                    dec_size=opt.attSize, generalized=False,feature_type = feature_type)

    acc = zsl_cls.acc
    per_acc = zsl_cls.per_acc

    if opt.unknown_classDistribution :
        zsl_cls = zsl_cls
        if opt.prior_estimation == 'BBSE':
            syn_feature, syn_label = generate_syn_feature(opt, netG, data.unseenclasses, data.attribute, opt.syn_num2)
            syn_feature2, syn_label2 = generate_syn_feature(opt, netG, data.unseenclasses,data.attribute, opt.syn_num2)
            lsp = ls(syn_feature, map_label(syn_label, data.unseenclasses),\
                syn_feature2, map_label(syn_label2, data.unseenclasses),data.test_unseen_feature,att_size=opt.attSize,nclass=len(data.unseenclasses),netR=netR,soft=opt.soft)
            w =lsp.predict_wt()
            w = np.squeeze(w)
            normalized_w = w / np.sum(w)
            class_prior_es = normalized_w
            logger.info(f'w_esimate:{w}')
        elif opt.prior_estimation == 'classifier':
            class_prior_es = zsl_cls.frequency
        elif opt.prior_estimation == 'CPE':
            from sklearn.cluster import KMeans
            support_center = np.array(zsl_cls.cls_center)
            kmeans = KMeans(n_clusters=len(data.unseenclasses),random_state=0,init= support_center).fit(zsl_cls.test_unseen_feature)
            las = kmeans.labels_
            frequency = np.bincount(las)/len(las)
            class_prior_es = frequency/frequency.sum()
                        
            # from visual import tsne_visual
            # center = kmeans.cluster_centers_
            # dd = np.concatenate([support_center,center])
            # tsne_visual(data.test_unseen_feature,las,path = 'support.pdf')
            # tsne_visual(unseen_data,data.test_unseen_label,path='gt.pdf')
        else:
            class_prior_es = class_prior
        class_prior = class_prior_es
        logger.info(f"Real Vs Estimated class prior ({opt.prior_estimation} strategy):\n{data.real_class_prior}\n{class_prior_es}")
    
    training_logger.append(['ZSL/acc'],[acc.item()],epoch)

    if best_zsl_acc < acc:
        best_zsl_acc = acc
        cur_path = f'{log_dir}/acc_{acc}.pth'
        save_dict = {'netG_state_dict':netG.state_dict(),
                    }
        if opt.R:
            save_dict['netR_state_dict'] = netR.state_dict()
        torch.save(save_dict,cur_path)
        if pre_path is not None:
            os.remove(pre_path)
        pre_path = cur_path
        
    logger.info(f'Epoch {epoch}: Current ZSL unseen accuracy={acc:.4f}')
    logger.info(f'the best ZSL unseen accuracy is {best_zsl_acc}')
    
    netG.train()
    if opt.R:
        netR.train()

training_logger.close()
