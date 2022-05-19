from functools import partial
import torch

from model.vit import VisionTransformer, interpolate_pos_embed
import torch.nn as nn
from model.model import ALIGN
from utils.data_loader import getDataLoader
from utils.utils import load, loadJson,set_global_random_seed
import torchvision.transforms as transforms
from tqdm import tqdm
import argparse




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
    data_json = loadJson("/home/tywang/myURE/text-align-aug/data/samples.json")
    loader = getDataLoader(data_json,img_transforms,True,32)
    net = ALIGN(args).to(device)
    for batch in loader:
        image = batch['images'].to(device)
        image_aug = batch['images_aug'].to(device)
        text = batch['captions']
        net.forward(text,image,image_aug)



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--bert_ckpt_path", type=str,default="/data/transformers/bert-base-uncased", help="as named")
    parser.add_argument("--max_len", type=int,default=32, help="as named")
    parser.add_argument("--cuda_index", type=int,default=1, help="as named")
    parser.add_argument("--seed", type=int, default=16, help="as named")
    parser.add_argument("--load_vision_ckpt", type=bool, default=True, help="as named")
    parser.add_argument("--temp", type=float, default=0.07, help="as named")
    args = parser.parse_args()
    main(args)
