from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import torchvision as tv

from PIL import Image
import numpy as np
import random
import os
from transformers import ViTFeatureExtractor, ViTModel
from transformers import BertTokenizer

from .utils import  read_json

MAX_DIM = 299


def under_max(image):
    if image.mode != 'RGB':
        image = image.convert("RGB")

    shape = np.array(image.size, dtype=np.float)
    long_dim = max(shape)
    scale = MAX_DIM / long_dim

    new_shape = (shape * scale).astype(int)
    image = image.resize(new_shape)

    return image


class RandomRotation:
    def __init__(self, angles=[0, 90, 180, 270]):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle, expand=True)


vit_train = tv.transforms.Compose([ 
        tv.transforms.RandomHorizontalFlip(),
        tv.transforms.ColorJitter(brightness=[0.5, 1.3], contrast=[
                                0.8, 1.5], saturation=[0.2, 1.5])
    ])
# vit_val  = None 不进行任何操作



def center_crop_val(image):
    h,w = image.size
    min_in_hw = max(h,w)
    img_transform_resize = tv.transforms.Compose([ 
        tv.transforms.CenterCrop(min_in_hw),
        tv.transforms.Resize([MAX_DIM,MAX_DIM]), 
        tv.transforms.ToTensor(),  
        tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    return img_transform_resize(image)



class CocoCaption(Dataset):
    def __init__(self,config, root, ann, max_length, limit, mode='training'):
        super().__init__()

        self.root = root
        self.annot = [(self._process(val['image_id']), val['caption'])
                      for val in ann['annotations']]
        if mode == 'validation':
            self.annot = self.annot
        if mode == 'training':
            self.annot = self.annot[: limit]
        self.transform_aug = tv.transforms.Compose([ 
                  tv.transforms.Grayscale(3), # 3 就是变成灰色
            ])
        # 加载图像的feature_extractor
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(config.backbone)
        print("feature_extractor:",self.feature_extractor)

        self.tokenizer = BertTokenizer.from_pretrained(config.pre_train_bert_path)  # , do_lower=True
        self.max_length = max_length + 1

    def _process(self, image_id):
        val = str(image_id).zfill(12)
        return val + '.jpg'

    def __len__(self):
        return len(self.annot)

    def __getitem__(self, idx):
        image_id, caption = self.annot[idx]
        image = Image.open(os.path.join(self.root, image_id)).convert('RGB')
        w,h = image.size
        min_shape = int(min(w,h)*0.25)  
        transform_aug_centerCrop = tv.transforms.Compose([ 
               tv.transforms.CenterCrop([min_shape,min_shape])
            ])
        image_aug = transform_aug_centerCrop(image)

        image = self.feature_extractor(image)['pixel_values'][0]
        image_aug = self.feature_extractor(image_aug)['pixel_values'][0]

        caption_encoded = self.tokenizer.encode_plus(
            caption, max_length=self.max_length, pad_to_max_length=True, return_attention_mask=True, return_token_type_ids=False, truncation=True)

        caption = np.array(caption_encoded['input_ids'])
        cap_mask = (
            1 - np.array(caption_encoded['attention_mask'])).astype(bool)  # padding mask


        return image,image_aug, caption, cap_mask,np.array(caption_encoded['attention_mask']) 


def build_dataset(config, mode='training'):
    if mode == 'training':
        train_dir = os.path.join(config.dir, 'train2017')
        train_file = os.path.join(
            config.dir, 'annotations', 'captions_train2017.json')
        data = CocoCaption(config,train_dir, read_json(
            train_file), max_length=config.max_position_embeddings, limit=config.limit,  mode='training')
        return data

    elif mode == 'validation':
        val_dir = os.path.join(config.dir, 'val2017')
        val_file = os.path.join(
            config.dir, 'annotations', 'captions_val2017.json')
        data = CocoCaption(config,val_dir, read_json(
            val_file), max_length=config.max_position_embeddings, limit=config.limit,  mode='validation')
        return data

    else:
        raise NotImplementedError(f"{mode} not supported")
