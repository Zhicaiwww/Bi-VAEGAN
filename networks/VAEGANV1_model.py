import torch.nn as nn
import torch
import torch.nn.functional as F

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class norm(nn.Module):
    def __init__(self,radius=1):
        super().__init__()
        self.radius = radius
    def forward(self,x):
        x = self.radius * x/ torch.norm(x,p=2,dim=-1,keepdim=True)
        return x  
        
class AttR(nn.Module):
    def __init__(self, opt):
        super(AttR, self).__init__()
        self.fc1 = nn.Linear(opt.resSize , 4096)
        self.fc3 = nn.Linear(4096, opt.attSize)
        self.L2norm = norm(1)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.hidden = None
        self.sigmoid = None
        self.lamb = 1.0
        self.apply(weights_init)
        if opt.att_criterian == 'W1':
            self.loss = self.reconstruct_W1_loss
        else:
            self.loss = self.reconstruct_mse_loss
            self.mse = nn.MSELoss()
    def forward_att(self, hidden, att=None):

        h = self.fc3(hidden)
        h = self.L2norm(h)
        self.att_out = h
        return h

    def forward(self, feat, att=None,onto=None):
        h = feat
        self.hidden = self.lrelu(self.fc1(h))
        h_att = self.forward_att(self.hidden)
        if self.train() and att is not None:
            W1_att = self.loss(h_att,att)
            recon_loss = self.lamb*W1_att 
            return recon_loss , h_att
        else:
            return h_att
    
    def reconstruct_mse_loss(self,h_att,att):
        mse = self.mse(h_att,att)
        mse = mse.sum()/att.size(0)
        return mse  

    def reconstruct_W1_loss(self,h_att,att):
        wt = (h_att-att).pow(2)
        wt /= wt.sum(1).sqrt().unsqueeze(1).expand(wt.size(0),wt.size(1))
        loss = wt * (h_att-att).abs()
        return loss.sum()/loss.size(0)

    def getLayersOutDet(self):
        #used at synthesis time for feature agumentation
        return self.hidden.detach()





class MLP_CRITIC(nn.Module):
    def __init__(self, opt): 
        super(MLP_CRITIC, self).__init__()
        self.fc1 = nn.Linear(opt.resSize + opt.attSize, 4096)
        self.fc2 = nn.Linear(4096, 1)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.apply(weights_init)

    def forward(self, x, att):
        h = torch.cat((x, att), 1) 
        h = self.lrelu(self.fc1(h))
        h = self.fc2(h)
        return h

class MLP_CRITIC_un(nn.Module):
    def __init__(self, opt): 
        super(MLP_CRITIC_un, self).__init__()
        self.fc1 = nn.Linear(opt.resSize , 4096)
        self.fc2 = nn.Linear(4096, 1)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.apply(weights_init)

    def forward(self, x):
        h = self.lrelu(self.fc1(x))
        h = self.fc2(h)
        return h

class netRCritic(nn.Module):
    def __init__(self, opt): 
        super(netRCritic, self).__init__()
        self.fc1 = nn.Linear(opt.attSize , 100)
        self.fc2 = nn.Linear(100, 1)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.apply(weights_init)

    def forward(self, x):
        h = self.lrelu(self.fc1(x))
        h = self.fc2(h)
        return h

class Decoder(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.norm =opt.L2_norm
        self.radius=opt.radius
        input_size = 2*opt.attSize 
        self.MLP = nn.Sequential(*[nn.Linear(input_size, 4096),
        
                                    nn.LeakyReLU(0.2,True),
                                    nn.Linear(4096,opt.resSize)]
                                    ) 
        self.apply(weights_init)

    def forward(self, z, c):
        z = torch.cat((z, c), dim=-1)
        out = self.MLP(z)
        self.out = out
        if self.norm:
            x =  self.radius * out / torch.norm(out,p=2,dim=1,keepdim=True)
        else:
            x = F.sigmoid(out)
        return x
    
    def get_out(self):
        return self.out
               
    def minmax(self, a):
        min_a = torch.min(a,dim=1,keepdim=True)[0]
        max_a = torch.max(a,dim=1,keepdim=True)[0]
        n2 = (a - min_a) / (max_a - min_a)
        return n2



class Encoder(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.MLP = nn.Sequential(*[nn.Linear(opt.resSize+opt.attSize,4096),
                                    nn.BatchNorm1d(4096),
                                    nn.LeakyReLU(0.2,True),])
        self.linear_means = nn.Linear(4096, opt.attSize)
        self.linear_log_var = nn.Linear(4096, opt.attSize)

    def forward(self, x, c=None):
        x = torch.cat((x, c), dim=-1)
        x = self.MLP(x)
        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)

        return means, log_vars


