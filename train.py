from functools import partial
import torch

from model.vit import VisionTransformer, interpolate_pos_embed
import torch.nn as nn
from utils.data_loader import getDataLoader
from utils.utils import load, loadJson
import torchvision.transforms as transforms
from tqdm import tqdm

device = torch.device("cuda:2")
data_json = loadJson("/home/tywang/myURE/text-align-aug/data/samples.json")
img_transforms = transforms.Compose([ 
    transforms.RandomHorizontalFlip(1), 
    transforms.Resize([256,256]), 
    transforms.ToTensor(),  
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])  
])
loader = getDataLoader(data_json,img_transforms,True,32)
batch = next(iter(loader))

visual_encoder = VisionTransformer(
            img_size=256, patch_size=16, embed_dim=768, depth=12, num_heads=12, 
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6)).to(device)
for batch in tqdm(loader):
    img_embed = visual_encoder(batch['images'].to(device))
    print(img_embed.shape)
    stop = 1
