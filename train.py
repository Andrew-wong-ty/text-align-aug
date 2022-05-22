import sys
import os
from pathlib import Path

CURR_FILE_PATH = (os.path.abspath(__file__))
PATH = Path(CURR_FILE_PATH)
CURR_DIR = str(PATH.parent.absolute())
sys.path.append(CURR_DIR)
P = PATH.parent
for i in range(3): # add parent path, height = 3
    P = P.parent
    sys.path.append(str(P.absolute()))
import time
TIME=time.strftime("%m-%d-%H*%M*%S", time.localtime())# 记录被初始化的时间
print("time",TIME)
import torch
from model.vit import VisionTransformer, interpolate_pos_embed
import torch.nn as nn
from torch import optim as optim
from model.model import ALIGN
from utils.data_loader import getDataLoader
from utils.utils import load, loadJson,set_global_random_seed,save
import torchvision.transforms as transforms
from tqdm import tqdm
import argparse
import numpy as np




def main(args):
    print(args)
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
    optimizer = optim.AdamW(net.parameters(), lr=args.lr)
    record_data = {
        "batch_CL_IT":[],
        "batch_W_flip":[],
        "batch_CEloss":[],
        "epoch_CL_IT":[],
        "epoch_W_flip":[],
        "epoch_CEloss":[],
    }
    for i in  range(args.epoch):
        closs_text_imgs = []
        closs_augs = []
        celosses = []
        n_step = len(loader)
        for i_batch, batch in enumerate(loader):
            image = batch['images'].to(device)
            image_aug = batch['images_aug'].to(device)
            text = batch['captions']
            closs_text_img,closs_aug,celoss = net.forward(text,image,image_aug)
            loss = closs_text_img+closs_aug+celoss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            closs_text_imgs.append(closs_text_img.detach().cpu().item())
            closs_augs.append(closs_aug.detach().cpu().item())
            celosses.append(celoss.detach().cpu().item())
            sys.stdout.write('\r')
            sys.stdout.write('Trian : [%3d/%3d]\t CLoss(IT): %.4f (%.4f)\t  CLoss(Wrot): %.4f (%.4f)\tCELoss: %.4f (%.4f)    total:%.4f (%.4f) '
                    %( i_batch+1, n_step, np.mean(closs_text_imgs),closs_text_img.data, 
                                          np.mean(closs_augs),closs_aug.data,
                                          np.mean(celosses),celoss.data,
                                          np.mean(closs_text_imgs)+  np.mean(closs_augs)+ np.mean(celosses), closs_text_imgs[-1]+closs_augs[-1]+celosses[-1]))
            sys.stdout.flush()
            record_data["batch_CL_IT"].append(closs_text_imgs[-1])
            record_data["batch_W_flip"].append(closs_augs[-1])
            record_data["batch_CEloss"].append(celosses[-1])
            #save(record_data,"/home/tywang/myURE/text-align-aug/logs/record/record_text_img_align_{}.pkl".format(TIME))
        record_data["epoch_CL_IT"].append(np.mean(closs_text_imgs))
        record_data["epoch_W_flip"].append(np.mean(closs_augs))
        record_data["epoch_CEloss"].append(np.mean(celosses))
       # save(record_data,"/home/tywang/myURE/text-align-aug/logs/record/record_text_img_align_{}.pkl".format(TIME))



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--bert_ckpt_path", type=str,default="/data/transformers/bert-base-uncased", help="as named")
    parser.add_argument("--max_len", type=int,default=32, help="as named")
    parser.add_argument("--lr", type=float,default=0.0001, help="as named")
    
    parser.add_argument("--cuda_index", type=int,default=3, help="as named")
    parser.add_argument("--epoch", type=int,default=100, help="as named")
    parser.add_argument("--seed", type=int, default=16, help="as named")
    parser.add_argument("--n_head", type=int, default=4, help="as named")
    parser.add_argument("--num_decoder_layers", type=int, default=8, help="as named")
    parser.add_argument("--load_vision_ckpt", type=bool, default=True, help="as named")
    parser.add_argument("--temp", type=float, default=0.9, help="less than 1.0")
    args = parser.parse_args()
    main(args)
# nohup python -u /home/tywang/myURE/text-align-aug/train.py >/home/tywang/myURE/text-align-aug/logs/record_text_img_align_fix_nor_0.0001.log 2>&1 &