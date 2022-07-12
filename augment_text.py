import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import BertTokenizer
from PIL import Image
from transformers import ViTFeatureExtractor
from sentence_transformers import SentenceTransformer
import argparse
import torchvision as tv
import  nltk.translate.bleu_score as bleu
from models import caption
from models.utils import save,load,find_first,create_caption_and_mask,process
from datasets import coco, utils
from configuration import Config
import os
import copy
import shutil
import json
from typing import List
from torchmetrics.text.bert import BERTScore
import numpy as np

os.environ['TOKENIZERS_PARALLELISM'] = 'false'



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
7.0 0.25  /data/tywang/vision_translation/catr_ckpt/ckpt_T07-01-12_23_26_epo9.pth
7. 0.5 纠正 /data/tywang/vision_translation/catr_ckpt/ckpt_T07-01-01_50_38_epo5.pth
8. 0.75纠正 /data/tywang/vision_translation/catr_ckpt/ckpt_T07-01-01_48_54_epo5.pth
9. 1.0  /data/tywang/vision_translation/catr_ckpt/ckpt_T07-01-12_22_34_epo6.pth
10. 一直res /data/tywang/vision_translation/catr_ckpt/ckpt_T07-05-10_05_00_bestModel.pth
11 2pretrain再加res /data/tywang/vision_translation/catr_ckpt/ckpt_T07-05-10_15_34_bestModel.pth
12. CL loss fixed bug: /data/tywang/vision_translation/catr_ckpt/ckpt_T07-09-11_48_16_bestModel.pth
13. cc12m res /data/tywang/vision_translation/catr_ckpt/ckpt_T07-10-12_36_34_bestModel.pth
14. cc12m no res /data/tywang/vision_translation/catr_ckpt/ckpt_T07-10-12_35_51_bestModel.pth

"""

parser = argparse.ArgumentParser(description='text augmentation via vision translation')
parser.add_argument('--checkpoint', type=str, help='checkpoint path', default="/data/tywang/vision_translation/catr_ckpt/ckpt_T07-10-12_36_34_bestModel.pth")
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
def get_augs_multiple_prompt(text:List[str],tokenizer):
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
    caption[:,:2] = caption_input[:,:2]  # 测试'prompt'
    cap_mask[:,:2] = False

    has_ended = torch.full((bs,), False).bool()
    for i in range(1,config.max_position_embeddings - 1):
        predictions = model.forward_predict_onStep(text_input=caption_input,
                                    text_input_attn_mask=caption_input_attn_mask,target=caption,target_mask=cap_mask,
                                    aug=False)
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
                                    text_input_attn_mask=caption_input_attn_mask,target=caption,target_mask=cap_mask,
                                    aug=False)
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
        "augmentation":[],
        "score":[]
    }
# using BERTscore for evaluation
bertscore = BERTScore(model_name_or_path="/data/transformers/bert-base-uncased",device=device)
batch_size = 64
start = 9999
end = start+64
index = list(range(start,end))
n_split = np.ceil(len(index)/batch_size).astype(int)
# get_augs_multiple_prompt
# get_augs_multiple
temp = get_augs_multiple(['We went to the lake, because a shark had been seen at the ocean beach, so it was a safer place to swim.',
                        "this new jangle of noise , mayhem and stupidity must be a serious contender for the title .",
                        "A former teammate , Carlton Dotson , has been charged with the murder . His body was found July 25 , and former teammate Carlton Dotson has been charged in his shooting death .",
                        "she is the mother.",
                        "A cat is playing a piano.",
                        "A group of people dance on a hill.",
                        "How can I improve my communication and verbal skills? ",
                        "Unlike other domestic species which were primarily selected for production-related traits, dogs were initially selected for their behaviors.",
                        " Construction of the present church began in 1245, on the orders of King Henry III.",
                        "The actress used to be named Terpsichore, but she changed it to Tina a few years ago, because she figured it was too hard to pronounce."
                        ]
                        
                        
                        ,tokenizer)
for item in temp:
    print("-> ",item,"\n")
# for i in tqdm(range(n_split)):
#     origin = [item['caption'] for item in annotations[start+i*batch_size:min(start+(i+1)*batch_size,end)]]
#     augmentations = get_augs_multiple(origin,tokenizer)
#     scores = bertscore(origin, augmentations)['f1']
#     generated_texts['score'].extend(scores)
#     generated_texts['origin'].extend(origin)
#     generated_texts['augmentation'].extend(augmentations)

# for idx, (ori,aug,sco) in enumerate(zip(generated_texts['origin'],generated_texts['augmentation'],generated_texts['score'] )):
#     print("| ",idx," | ",ori," | ",aug.capitalize()," |",sco)
# print("avg_score: ",np.mean(generated_texts['score']))
#save(generated_texts,"/home/tywang/myURE/vision_translate_align_bert/save_data/aug_710_CL_S{:.3f}.pkl".format(np.mean(generated_texts['score'])))





















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






