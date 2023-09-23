import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader

from modules import Generator, Gaussian_Predictor, Decoder_Fusion, Label_Encoder, RGB_Encoder

from dataloader import Dataset_Dance
from torchvision.utils import save_image
import random
import torch.optim as optim
from torch import stack

from tqdm import tqdm
import imageio

import matplotlib.pyplot as plt
from math import log10

def Generate_PSNR(imgs1, imgs2, data_range=1.):
    """PSNR for torch tensor"""
    mse = nn.functional.mse_loss(imgs1, imgs2, reduction = 'mean') 
    psnr = 20 * log10(data_range) - 10 * torch.log10(mse)
    return psnr


def kl_criterion(mu, logvar, batch_size):
  KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
  KLD /= batch_size  
  return KLD


class kl_annealing():
    def __init__(self, args, current_epoch=0):
        self.cur = 0
        self.type = args.kl_anneal_type
        self.cycle = args.kl_anneal_cycle
        self.ratio = args.kl_anneal_ratio
        self.total_iter = args.num_epoch * args.train_vi_len
        self.beta_list = np.ones(self.total_iter)

        if(self.type == 'Monotonic'):
            self.cycle = 1
            self.frange_cycle_linear(self.total_iter, n_cycle = self.cycle, ratio = 0.25)

        elif(self.type == 'Cyclical'):
            self.frange_cycle_linear(self.total_iter, n_cycle = self.cycle, ratio = self.ratio)

        elif(self.type == 'None'):
            self.beta_list = np.zeros(self.total_iter)

    def update(self):
        self.cur += 1
    
    def get_beta(self):
        return self.beta_list[self.cur]

    def frange_cycle_linear(self, n_iter, start=0.0, stop=1.0,  n_cycle=1, ratio=1):
        times = n_iter / n_cycle
        step = (stop - start) / (times * ratio)

        for c in range(n_cycle):
            s, i = start, 0
            while s <= stop and (int(i + c * times) < n_iter):
                self.beta_list[int(i + c * times)] = s
                s += step
                i += 1
        

class VAE_Model(nn.Module):
    def __init__(self, args):
        super(VAE_Model, self).__init__()
        self.args = args
        self.loss_list = []
        self.tfr_list = []
        self.PSNR_list = []
        # Modules to transform image from RGB-domain to feature-domain
        self.frame_transformation = RGB_Encoder(3, args.F_dim)
        self.label_transformation = Label_Encoder(3, args.L_dim)
        
        # Conduct Posterior prediction in Encoder
        self.Gaussian_Predictor   = Gaussian_Predictor(args.F_dim + args.L_dim, args.N_dim)
        self.Decoder_Fusion       = Decoder_Fusion(args.F_dim + args.L_dim + args.N_dim, args.D_out_dim)
        
        # Generative model
        self.Generator            = Generator(input_nc=args.D_out_dim, output_nc=3)
        
        self.optim      = optim.Adam(self.parameters(), lr=self.args.lr)
        self.scheduler  = optim.lr_scheduler.MultiStepLR(self.optim, milestones=[2, 5], gamma=0.1)
        self.kl_annealing = kl_annealing(args, current_epoch=0)
        self.mse_criterion = nn.MSELoss()
        self.current_epoch = 0
        
        # Teacher forcing arguments
        self.tfr = args.tfr
        self.tfr_d_step = args.tfr_d_step
        self.tfr_sde = args.tfr_sde
        
        self.train_vi_len = args.train_vi_len
        self.val_vi_len   = args.val_vi_len
        self.batch_size = args.batch_size
        
        
    def forward(self, img, label):
        pass
    
    def training_stage(self):
        for i in range(self.args.num_epoch):
            train_loader = self.train_dataloader()
            adapt_TeacherForcing = True if random.random() < self.tfr else False
            self.tfr_list.append(self.tfr)
            for (img, label) in (pbar := tqdm(train_loader, ncols=120)):
                
                img = img.to(self.args.device)
                label = label.to(self.args.device)
                loss = self.training_one_step(img, label, adapt_TeacherForcing)
                self.loss_list.append(loss)
                beta = self.kl_annealing.get_beta()
                if adapt_TeacherForcing:
                    self.tqdm_bar('train [TeacherForcing: ON, {:.1f}], beta: {}'.format(self.tfr, beta), pbar, loss.detach().cpu(), lr=self.scheduler.get_last_lr()[0])
                else:
                    self.tqdm_bar('train [TeacherForcing: OFF, {:.1f}], beta: {}'.format(self.tfr, beta), pbar, loss.detach().cpu(), lr=self.scheduler.get_last_lr()[0])
            
            if self.current_epoch % self.args.per_save == 0:
                self.save(os.path.join(self.args.save_root, f"epoch={self.current_epoch}.ckpt"))
                
            self.eval()
            self.current_epoch += 1
            self.scheduler.step()
            self.teacher_forcing_ratio_update()
            self.kl_annealing.update()
        self.loss_plot()
        self.tfr_plot() 
            
    @torch.no_grad()
    def eval(self):
        val_loader = self.val_dataloader()
        for (img, label) in (pbar := tqdm(val_loader, ncols=120)):
            img = img.to(self.args.device)
            label = label.to(self.args.device)
            loss = self.val_one_step(img, label)
            self.tqdm_bar('val', pbar, loss.detach().cpu(), lr=self.scheduler.get_last_lr()[0])
    
    def training_one_step(self, img, label, adapt_TeacherForcing):
        self.Generator.zero_grad()
        self.optim.zero_grad()
        self.Gaussian_Predictor.zero_grad() 
        self.Decoder_Fusion.zero_grad() 
        self.label_transformation.zero_grad() 
        self.frame_transformation.zero_grad()
        
        img = img.permute(1, 0, 2, 3, 4) # change tensor into (seq, B, C, H, W)
        label = label.permute(1, 0, 2, 3, 4) # change tensor into (seq, B, C, H, W)

        # Normal normal
        last_human_feat = self.frame_transformation(img[0])
        first_templete = last_human_feat.clone()
        out = img[0]
        mseLoss = 0
        kl = 0

        for i in range(1, self.train_vi_len):
            z = torch.cuda.FloatTensor(1, self.args.N_dim, self.args.frame_H, self.args.frame_W).normal_()
            label_feat = self.label_transformation(label[i])
            human_feat_hat = self.frame_transformation(out)
            prev = self.frame_transformation(img[i-1])
            gt = self.frame_transformation(img[i])
            if adapt_TeacherForcing:
                parm = self.Decoder_Fusion(prev, label_feat, z)    
            else:
                parm = self.Decoder_Fusion(human_feat_hat, label_feat, z)    
            out = self.Generator(parm)
            
            _, mu, logvar = self.Gaussian_Predictor(gt, label_feat)
            kl += kl_criterion(mu, logvar, self.batch_size)
            mseLoss += self.mse_criterion(out, img[i])

        beta = self.kl_annealing.get_beta()
        loss = mseLoss + kl * beta
        loss = loss / (self.train_vi_len - 1)
        print(f"loss:{loss:.2f}")
        loss.backward()
        self.optim.step()
        return loss 

    def val_one_step(self, img, label):
        img = img.permute(1, 0, 2, 3, 4) # change tensor into (seq, B, C, H, W)
        label = label.permute(1, 0, 2, 3, 4) # change tensor into (seq, B, C, H, W)
        self.PSNR_list = []
        decoded_frame_list = [img[0].cpu()]
        img_list = []
        # Normal normal
        last_human_feat = self.frame_transformation(img[0])
        first_templete = last_human_feat.clone()
        out = img[0]
        mseLoss = 0
        kl = 0

        for i in range(1, self.val_vi_len):
            z = torch.cuda.FloatTensor(1, self.args.N_dim, self.args.frame_H, self.args.frame_W).normal_()
            label_feat = self.label_transformation(label[i])
            human_feat_hat = self.frame_transformation(out)
            gt = self.frame_transformation(img[i])
            parm = self.Decoder_Fusion(human_feat_hat, label_feat, z)    
            out = self.Generator(parm)
            
            decoded_frame_list.append(out.cpu())
            img_list.append(img[i].cpu())

            _, mu, logvar = self.Gaussian_Predictor(gt, label_feat)
            kl += kl_criterion(mu, logvar, self.batch_size)
            mseLoss += self.mse_criterion(out, img[i])
        beta = self.kl_annealing.get_beta()
        loss = mseLoss + kl * beta
        loss = loss / (self.val_vi_len - 1)
        print(f"loss:{loss:.2f}")
        
        decoded_frame_list = decoded_frame_list[1:]

        generated_frame = stack(decoded_frame_list).permute(1, 0, 2, 3, 4)
        img_frame = stack(img_list).permute(1, 0, 2, 3, 4)

        os.makedirs("./validation_frame", exist_ok=True)
        for i in range(629):
            PSNR = Generate_PSNR(generated_frame[0][i], img_frame[0][i])
            self.PSNR_list.append(PSNR.item())

        self.make_gif(generated_frame[0], "./validation_frame/pred_seq.gif")
        self.make_gif(img_frame[0], "./validation_frame/pose.gif")
        self.PSNR_plot()
        return loss 
                
    def make_gif(self, images_list, img_name):
        new_list = []
        for img in images_list:
            new_list.append(transforms.ToPILImage()(img))
            
        new_list[0].save(img_name, format="GIF", append_images=new_list,
                    save_all=True, duration=40, loop=0)
    
    def train_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize((self.args.frame_H, self.args.frame_W)),
            transforms.ToTensor()
        ])

        dataset = Dataset_Dance(root=self.args.DR, transform=transform, mode='train', video_len=self.train_vi_len, \
                                                partial=args.fast_partial if self.args.fast_train else args.partial)
        if self.current_epoch > self.args.fast_train_epoch:
            self.args.fast_train = False
            
        train_loader = DataLoader(dataset,
                                  batch_size=self.batch_size,
                                  num_workers=self.args.num_workers,
                                  drop_last=True,
                                  shuffle=False)  
        return train_loader
    
    def val_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize((self.args.frame_H, self.args.frame_W)),
            transforms.ToTensor()
        ])
        dataset = Dataset_Dance(root=self.args.DR, transform=transform, mode='val', video_len=self.val_vi_len, partial=1.0)  
        val_loader = DataLoader(dataset,
                                  batch_size=1,
                                  num_workers=self.args.num_workers,
                                  drop_last=True,
                                  shuffle=False)  
        return val_loader
    
    def teacher_forcing_ratio_update(self):
        self.tfr *= (1 - self.tfr_d_step)
        self.tfr = max(0, self.tfr)
            
    def tqdm_bar(self, mode, pbar, loss, lr):
        pbar.set_description(f"({mode}) Epoch {self.current_epoch}, lr:{lr}" , refresh=False)
        pbar.set_postfix(loss=float(loss), refresh=False)
        pbar.refresh()
        
    def save(self, path):
        torch.save({
            "state_dict": self.state_dict(),
            "optimizer": self.state_dict(),  
            "lr"        : self.scheduler.get_last_lr()[0],
            "tfr"       :   self.tfr,
            "last_epoch": self.current_epoch
        }, path)
        print(f"save ckpt to {path}")

    def load_checkpoint(self):
        if self.args.ckpt_path != None:
            checkpoint = torch.load(self.args.ckpt_path)
            self.load_state_dict(checkpoint['state_dict'], strict=True) 
            self.args.lr = checkpoint['lr']
            self.tfr = checkpoint['tfr']
            
            self.optim      = optim.Adam(self.parameters(), lr=self.args.lr)
            self.scheduler  = optim.lr_scheduler.MultiStepLR(self.optim, milestones=[2, 4], gamma=0.1)
            self.kl_annealing = kl_annealing(self.args, current_epoch=checkpoint['last_epoch'])
            self.current_epoch = checkpoint['last_epoch']

    def optimizer_step(self):
        nn.utils.clip_grad_norm_(self.parameters(), 1.)
        self.optim.step()

    def loss_plot(self):
        plt.clf() 
        plt.xlabel('itration')
        plt.ylabel('Loss')
        plt.title('Loss Curve')
        loss_list_cpu = [loss.item() for loss in self.loss_list]
        # plt.plot(loss_list_cpu, label = 'monotonic', color = 'y')
        plt.plot(loss_list_cpu, label = 'cyclical', color = 'b')
        # plt.plot(loss_list_cpu, label = 'None', color = 'g')
        plt.legend()
        plt.savefig('loss_plot.png', format='png')
        # plt.show()
        
    def tfr_plot(self):
        plt.clf() 
        plt.xlabel('Epoch')
        plt.ylabel('teacher_forcing_ratio')
        plt.title('tfr Curve')
        plt.plot(self.tfr_list, label = 'tfr', color = 'b')
        plt.legend()
        plt.savefig('tfr_plot.png', format='png')
        # plt.show()

    def PSNR_plot(self):
        plt.clf() 
        plt.xlabel('Frame index')
        plt.ylabel('PSNR')
        plt.title('Per frame Quality (PSNR)')
        plt.plot(self.PSNR_list, label = 'PSNR', color = 'b')
        plt.legend()
        plt.savefig('PSNR_plot.png', format='png')
        # plt.show()

def main(args):
    os.makedirs(args.save_root, exist_ok=True)
    model = VAE_Model(args).to(args.device)
    model.load_checkpoint()
    if args.test:
        model.eval()
    else:
        model.training_stage()
    # model.loss_plot()
    # model.tfr_plot()
    # model.PSNR_plot()




if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--batch_size',    type=int,    default=2)
    parser.add_argument('--lr',            type=float,  default=0.001,     help="initial learning rate")
    parser.add_argument('--device',        type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument('--optim',         type=str, choices=["Adam", "AdamW"], default="Adam")
    parser.add_argument('--gpu',           type=int, default=1)
    parser.add_argument('--test',          action='store_true')
    parser.add_argument('--store_visualization',      action='store_true', help="If you want to see the result while training")
    parser.add_argument('--DR',            type=str, required=True,  help="Your Dataset Path")
    parser.add_argument('--save_root',     type=str, required=True,  help="The path to save your data")
    parser.add_argument('--num_workers',   type=int, default=4)
    parser.add_argument('--num_epoch',     type=int, default=70,     help="number of total epoch")
    parser.add_argument('--per_save',      type=int, default=1,      help="Save checkpoint every seted epoch")
    parser.add_argument('--partial',       type=float, default=1.0,  help="Part of the training dataset to be trained")
    parser.add_argument('--train_vi_len',  type=int, default=16,     help="Training video length")
    parser.add_argument('--val_vi_len',    type=int, default=630,    help="valdation video length")
    parser.add_argument('--frame_H',       type=int, default=32,     help="Height input image to be resize")
    parser.add_argument('--frame_W',       type=int, default=64,     help="Width input image to be resize")
    
    
    # Module parameters setting     
    parser.add_argument('--F_dim',         type=int, default=128,    help="Dimension of feature human frame")
    parser.add_argument('--L_dim',         type=int, default=32,     help="Dimension of feature label frame")
    parser.add_argument('--N_dim',         type=int, default=12,     help="Dimension of the Noise")
    parser.add_argument('--D_out_dim',     type=int, default=192,    help="Dimension of the output in Decoder_Fusion")
    
    # Teacher Forcing strategy
    parser.add_argument('--tfr',           type=float, default=1.0,  help="The initial teacher forcing ratio")
    parser.add_argument('--tfr_sde',       type=int,   default=10,   help="The epoch that teacher forcing ratio start to decay")
    parser.add_argument('--tfr_d_step',    type=float, default=0.1,  help="Decay step that teacher forcing ratio adopted")
    parser.add_argument('--ckpt_path',     type=str,    default=None,help="The path of your checkpoints")   
    
    # Training Strategy
    parser.add_argument('--fast_train',         action='store_true')
    parser.add_argument('--fast_partial',       type=float, default=0.4,    help="Use part of the training data to fasten the convergence")
    parser.add_argument('--fast_train_epoch',   type=int, default=5,        help="Number of epoch to use fast train mode")
    
    # Kl annealing stratedy arguments
    parser.add_argument('--kl_anneal_type',     type=str, default='Cyclical',       help="")
    parser.add_argument('--kl_anneal_cycle',    type=int, default=10,               help="")
    parser.add_argument('--kl_anneal_ratio',    type=float, default=1,              help="")
    

    

    args = parser.parse_args()
    
    main(args)
