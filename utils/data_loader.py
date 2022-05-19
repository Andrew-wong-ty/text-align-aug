import sys
import os
from pathlib import Path

CURR_FILE_PATH = (os.path.abspath(__file__))
PATH = Path(CURR_FILE_PATH)
CURR_DIR = str(PATH.parent.absolute())
sys.path.append(CURR_DIR)
P = PATH.parent
for i in range(3): 
    P = P.parent
    sys.path.append(str(P.absolute()))

import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import pickle
import numpy as np
import nltk
from PIL import Image

class textImageDataset(data.Dataset):
    """Customized Datset with torch.utils.dara.DataLoader"""
    def __init__(self,dataJson,transform):
        """Set the path for images
        
        Args:
            `dataJson`: 
                a list like [{'image':xxx.jpg, 'caption':xxx},....{}]
                where 'image' is the path of the image and caption is the corresponding caption.
            `transform`:
                image transformer
        """
        self.dataJson = dataJson
        self.transform = transform
        self.common_transoform = transforms.Compose([ 
                                transforms.Resize([256,256]), 
                                transforms.ToTensor(),  
                                transforms.Normalize([0.485,0.456,0.406],
                                                    [0.229,0.224,0.225])  
                            ])
    def __getitem__(self, index):
        image_path = self.dataJson[index]['image']
        image = Image.open(image_path).convert('RGB')
        image_aug = self.transform(image)
        image_normal = self.common_transoform(image)
        text = self.dataJson[index]['caption']
        return {'images':image_normal,'images_aug':image_aug, 'captions':text}
    def __len__(self):
        return len(self.dataJson)


def getDataLoader(data_json, transform, shuffle, batch_size):
    """return the data loader from textImageDataset"""
    dataset = textImageDataset(dataJson=data_json,transform=transform)
    data_loader = torch.utils.data.DataLoader(dataset=dataset, 
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=0)
    return data_loader
        

