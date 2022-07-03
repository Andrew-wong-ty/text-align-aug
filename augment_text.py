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
from models.utils import save,load,find_first,create_caption_and_mask,process
from datasets import coco, utils
from configuration import Config
import os
import copy
import shutil
import json
from typing import List



parser = argparse.ArgumentParser(description='text augmentation via vision translation')
parser.add_argument('--checkpoint', type=str, help='checkpoint path', default="/data/tywang/vision_translation/catr_ckpt/ckpt_T07-01-01_48_54_epo5.pth")
parser.add_argument('--coco_val_path', type=str, help='the path of captions_val2017.json', default="/home/tywang/myURE/text-align-aug/data/annotations/captions_val2017.json")
args = parser.parse_args()
checkpoint_path = args.checkpoint
coco_val_path = args.coco_val_path

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
    model.eval()
    tokenizer = BertTokenizer.from_pretrained(config.pre_train_bert_path)



@torch.no_grad()
def get_augs_multiple(text:List[str],tokenizer):
    # setup tokenizer
    start_token = tokenizer.convert_tokens_to_ids(tokenizer._cls_token)
    end_token = tokenizer.convert_tokens_to_ids(tokenizer._sep_token)
    pad_token = tokenizer.convert_tokens_to_ids(tokenizer._pad_token)

    # prepare input
    bs = len(text)
    text_input = tokenizer.batch_encode_plus(text,max_length=config.max_position_embeddings, 
                    pad_to_max_length=True, return_attention_mask=True, return_token_type_ids=False, truncation=True,return_tensors='pt')
    caption_input = text_input['input_ids'].to(device)
    caption_input_attn_mask = text_input['attention_mask'].to(device)

    # create input caption
    caption, cap_mask = create_caption_and_mask(start_token, config.max_position_embeddings, bs=bs)
    caption = caption.to(device)
    cap_mask = cap_mask.to(device)

    has_ended = torch.full((bs,), False).bool()
    for i in range(config.max_position_embeddings - 1):
        predictions = model.forward_predict_onStep(text_input=caption_input,
                                    text_input_attn_mask=caption_input_attn_mask,target=caption,target_mask=cap_mask,aug=True)
        predictions = predictions[:, i, :]
        predicted_id = torch.argmax(predictions, axis=-1)
        has_ended |= (predicted_id == end_token).type_as(has_ended)
        caption[:, i+1] = predicted_id
        cap_mask[:, i+1] = False

        if torch.all(has_ended):
            break
    eos_positions = find_first(caption, end_token)
    for i in range(bs):
        j = int(eos_positions[i].item()) + 1
        caption[i, j:] = pad_token
    res = tokenizer.batch_decode(caption.tolist(), skip_special_tokens=True)
    return res
    

# 打开captions_val2017.json, 获取里面的caption的augmentation
with open(coco_val_path,'r') as file:
    data = json.load(file)
    annotations = data['annotations']

generated_texts = {
        "origin":[],
        "augmentation":[]
    }

generated_texts['origin'] = [item['caption'] for item in annotations[9999:9999+50]]
generated_texts['augmentation'] = get_augs_multiple(generated_texts['origin'],tokenizer)

for idx, (ori,aug) in enumerate(zip(generated_texts['origin'],generated_texts['augmentation'])):
    print("| ",idx," | ",ori," | ",aug.capitalize()," |")

# save(generated_texts,"/home/tywang/myURE/vision_translate_align_bert/save_data/aug_701_Dual.pkl")





















################################################# deprecated  ################################################# 

# @torch.no_grad()
# def get_augs(text,info=""):
    

#     text_input = tokenizer.batch_encode_plus([text],max_length=config.max_position_embeddings, 
#                     pad_to_max_length=True, return_attention_mask=True, return_token_type_ids=False, truncation=True,return_tensors='pt')
#     caption_input = text_input['input_ids'].to(device)
#     caption_input_attn_mask = text_input['attention_mask'].to(device)

#     caption, cap_mask = create_caption_and_mask(
#         start_token, config.max_position_embeddings)
#     atten_mask = (~cap_mask).long().to(device)


#     for i in range(config.max_position_embeddings - 1):
#         """选择不同的方式得到predictions"""
#         # predictions,img_feature, text_feature =\
#         #      model(image,image_aug, caption, cap_mask, atten_mask)  # [1,128,30522] 这个已经和maxlen 绑定了, 但是自己的代码里面, 里面的pad都去掉了
#         # print(metric(img_feature,text_feature))    
        
#         predictions = model.forward_predict_onStep(text_input=caption_input,
#                                     text_input_attn_mask=caption_input_attn_mask,target=caption,target_mask=cap_mask,aug=True)
#         # 
#         """#########################"""
        
        
        
#         predictions = predictions[:, i, :]
#         predicted_id = torch.argmax(predictions, axis=-1)

#         if predicted_id[0] == 102:
#             break

#         caption[:, i+1] = predicted_id[0]
#         cap_mask[:, i+1] = False
#         atten_mask[:, i+1] = 1
#     caption = caption.detach().cpu()
#     result = tokenizer.decode(caption[0].tolist(), skip_special_tokens=True)
#     print("| ",info," | ",text," | ",result.capitalize()," |")
#     return result






