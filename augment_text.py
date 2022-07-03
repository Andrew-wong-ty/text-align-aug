import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import BertTokenizer
from PIL import Image
from transformers import ViTFeatureExtractor
from sentence_transformers import SentenceTransformer
import argparse
import torchvision as tv

from models import caption
from models.utils import save,load
from datasets import coco, utils
from configuration import Config
import os
import copy
import shutil
import json

"""
一些模型
1. fix住了VIT, 使用MSE作为loss的模型:  /data/tywang/vision_translation/catr_ckpt/ckpt_T06-27-11_44_41_epo7.pth
1.1 fix VIT, 使用CLoss /data/tywang/vision_translation/catr_ckpt/ckpt_T06-28-11_53_21_epo7.pth
2. 没fixVIT, 使用MSE作为对齐loss的模型
3. 没fix VIT, 使用CLoss作为对齐loss的模型
4. 全程: 黑白aug: /data/tywang/vision_translation/catr_ckpt/ckpt_T06-29-15_34_38_epo4.pth
4.1 全程: 中心cap: /data/tywang/vision_translation/catr_ckpt/ckpt_T06-29-15_40_00_epo4.pth
5. dual: /data/tywang/vision_translation/catr_ckpt/ckpt_T06-30-12_06_08_epo8.pth
6. MAE: /data/tywang/vision_translation/catr_ckpt/ckpt_T06-30-12_17_26_epo5.pth
7.0 0.25  /data/tywang/vision_translation/catr_ckpt/ckpt_T07-01-12_23_26_epo5.pth
7. 0.5 纠正 /data/tywang/vision_translation/catr_ckpt/ckpt_T07-01-01_50_38_epo5.pth
8. 0.75纠正 /data/tywang/vision_translation/catr_ckpt/ckpt_T07-01-01_48_54_epo5.pth
9. 1.0  /data/tywang/vision_translation/catr_ckpt/ckpt_T07-01-12_22_34_epo5.pth

"""

parser = argparse.ArgumentParser(description='Image Captioning')
parser.add_argument('--path', type=str, help='path to image', default="/home/tywang/myURE/image_samples/black.jpg")  # women_ski  group_people
parser.add_argument('--checkpoint', type=str, help='checkpoint path', default="/data/tywang/vision_translation/catr_ckpt/ckpt_T06-30-12_06_08_epo8.pth")
args = parser.parse_args()
image_path = args.path
checkpoint_path = args.checkpoint

config = Config()
device = torch.device("cuda:0")
######################加载模型
print("Checking for checkpoint.")
if checkpoint_path is None:
    raise NotImplementedError('No model to chose from!')
else:
    if not os.path.exists(checkpoint_path):
        raise NotImplementedError('Give valid checkpoint path')
    print("Found checkpoint! Loading!")
    model,_ ,_= caption.build_model(config)
    print("Loading Checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model = model.to(device)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

start_token = tokenizer.convert_tokens_to_ids(tokenizer._cls_token)
end_token = tokenizer.convert_tokens_to_ids(tokenizer._sep_token)




def create_caption_and_mask(start_token, max_length):
    caption_template = torch.zeros((1, max_length), dtype=torch.long)
    mask_template = torch.ones((1, max_length), dtype=torch.bool)  

    caption_template[:, 0] = start_token
    mask_template[:, 0] = False  

    return caption_template.to(device), mask_template.to(device)

def process(image_id):
    # 根据caption的图像id得到图像path
    val = str(image_id).zfill(12)
    return val + '.jpg'

model.eval()
@torch.no_grad()
def get_augs(text,info=""):
    

    text_input = tokenizer.batch_encode_plus([text],max_length=config.max_position_embeddings, 
                    pad_to_max_length=True, return_attention_mask=True, return_token_type_ids=False, truncation=True,return_tensors='pt')
    caption_input = text_input['input_ids'].to(device)
    caption_input_attn_mask = text_input['attention_mask'].to(device)

    caption, cap_mask = create_caption_and_mask(
        start_token, config.max_position_embeddings)
    atten_mask = (~cap_mask).long().to(device)


    for i in range(config.max_position_embeddings - 1):
        """选择不同的方式得到predictions"""
        # predictions,img_feature, text_feature =\
        #      model(image,image_aug, caption, cap_mask, atten_mask)  # [1,128,30522] 这个已经和maxlen 绑定了, 但是自己的代码里面, 里面的pad都去掉了
        # print(metric(img_feature,text_feature))    
        
        predictions = model.forward_predict_onStep(text_input=caption_input,
                                    text_input_attn_mask=caption_input_attn_mask,target=caption,target_mask=cap_mask,aug=True)
        # 
        """#########################"""
        
        
        
        predictions = predictions[:, i, :]
        predicted_id = torch.argmax(predictions, axis=-1)

        if predicted_id[0] == 102:
            break

        caption[:, i+1] = predicted_id[0]
        cap_mask[:, i+1] = False
        atten_mask[:, i+1] = 1
    caption = caption.detach().cpu()
    result = tokenizer.decode(caption[0].tolist(), skip_special_tokens=True)
    print("| ",info," | ",text," | ",result.capitalize()," |")
    return result

# 打开captions_val2017.json, 获取里面的caption的augmentation
with open("/home/tywang/myURE/text-align-aug/data/annotations/captions_val2017.json",'r') as file:
    data = json.load(file)
    annotations = data['annotations']

generated_texts = {
        "origin":[],
        "augmentation":[]
    }

for idx,item in enumerate(annotations[9999:9999+50]):
    text = item['caption']
    aug_text = get_augs(text,str(idx))
    generated_texts['origin'].append(text)
    generated_texts['augmentation'].append(aug_text)

# save(generated_texts,"/home/tywang/myURE/vision_translate_align_bert/save_data/aug_701_Dual.pkl")


