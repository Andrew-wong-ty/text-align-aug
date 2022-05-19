from functools import partial
import torch
import sys
from model.vit import VisionTransformer, interpolate_pos_embed
import torch.nn as nn
from torch import optim as optim
from model.model import ALIGN
from utils.data_loader import getDataLoader
from utils.utils import load, loadJson,set_global_random_seed
import torchvision.transforms as transforms
from tqdm import tqdm
import argparse
import numpy as np




def main(args):
    set_global_random_seed(args.seed)
    args.device = device = torch.device("cuda:{}".format(args.cuda_index))
    img_transforms = transforms.Compose([ 
        transforms.RandomHorizontalFlip(1), # 100% flip
        transforms.Resize([256,256]), 
        transforms.ToTensor(),  
        transforms.Normalize([0.485,0.456,0.406],
                            [0.229,0.224,0.225])  
    ])
    data_json = loadJson("/home/tywang/myURE/text-align-aug/data/samples_50000.json")
    loader = getDataLoader(data_json,img_transforms,True,32)
    net = ALIGN(args).to(device)
    optimizer = optim.AdamW(net.parameters(), lr=0.0001)
    for i in range(args.epoch):
        closs_text_imgs = []
        closs_augs = []
        n_step = len(loader)
        for i_batch, batch in enumerate(loader):
            image = batch['images'].to(device)
            image_aug = batch['images_aug'].to(device)
            text = batch['captions']
            closs_text_img,closs_aug = net.forward(text,image,image_aug)
            loss = closs_text_img+closs_aug
            loss.backward()
            optimizer.zero_grad()
            optimizer.step()
            closs_text_imgs.append(closs_text_img.detach().cpu().item())
            closs_augs.append(closs_aug.detach().cpu().item())
            sys.stdout.write('\r')
            sys.stdout.write('Trian : [%3d/%3d]\t CLoss(IT): %.4f (%.4f)\t  CLoss(Wrot): %.4f (%.4f)'
                    %( i_batch+1, n_step, np.mean(closs_text_imgs),closs_text_img.data, np.mean(closs_augs),closs_aug.data))
            sys.stdout.flush()
        



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--bert_ckpt_path", type=str,default="/data/transformers/bert-base-uncased", help="as named")
    parser.add_argument("--max_len", type=int,default=32, help="as named")
    parser.add_argument("--cuda_index", type=int,default=1, help="as named")
    parser.add_argument("--epoch", type=int,default=10, help="as named")
    parser.add_argument("--seed", type=int, default=16, help="as named")
    parser.add_argument("--load_vision_ckpt", type=bool, default=True, help="as named")
    parser.add_argument("--temp", type=float, default=2, help="as named")
    args = parser.parse_args()
    main(args)
