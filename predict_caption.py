import torch
import torch.nn as nn
from transformers import BertTokenizer
from PIL import Image
from transformers import ViTFeatureExtractor
from sentence_transformers import SentenceTransformer
import argparse
import torchvision as tv

from models import caption
from datasets import coco, utils
from configuration import Config
import os
import copy

"""
一些模型
1. fix住了VIT, 使用MSE作为loss的模型:  /data/tywang/vision_translation/catr_ckpt/ckpt_T06-27-11_44_41_epo7.pth
2. 没fixVIT, 使用MSE作为对齐loss的模型  /data/tywang/vision_translation/catr_ckpt/ckpt_T06-27-00_54_09_epo5.pth
3. 没fix VIT, 使用CLoss作为对齐loss的模型
4. fix了VIT, 使用CLoss 作为loss的模型  

"""

parser = argparse.ArgumentParser(description='Image Captioning')
parser.add_argument('--path', type=str, help='path to image', default="/home/tywang/myURE/image_samples/women_ski.jpg")  # women_ski  group_people
parser.add_argument('--v', type=str, help='version', default='v3(eliminate this)')
parser.add_argument('--checkpoint', type=str, help='checkpoint path', default="/data/tywang/vision_translation/catr_ckpt/ckpt_T06-28-11_53_21_epo7.pth")
args = parser.parse_args()
image_path = args.path
version = args.v
checkpoint_path = args.checkpoint

config = Config()

if version == 'v1':
    model = torch.hub.load('saahiluppal/catr', 'v1', pretrained=True)
elif version == 'v2':
    model = torch.hub.load('saahiluppal/catr', 'v2', pretrained=True)
elif version == 'v3':
    model = torch.hub.load('saahiluppal/catr', 'v3', pretrained=True)
else:
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
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

start_token = tokenizer.convert_tokens_to_ids(tokenizer._cls_token)
end_token = tokenizer.convert_tokens_to_ids(tokenizer._sep_token)

# 指定一个text
text = "A crowded city street filled with traffic and bicycles."
text_input = tokenizer.batch_encode_plus([text],max_length=config.max_position_embeddings, 
                pad_to_max_length=True, return_attention_mask=True, return_token_type_ids=False, truncation=True,return_tensors='pt')
caption_input = text_input['input_ids']
caption_input_attn_mask = text_input['attention_mask']
caption_input_pad_mask = (1-caption_input_attn_mask).bool()

transform_aug = tv.transforms.Compose([ 
                 tv.transforms.RandomHorizontalFlip(1)
            ])
feature_extractor = ViTFeatureExtractor.from_pretrained('/data/tywang/vision_transformer/google_vit-base-patch16-224')
image = Image.open(image_path)
image_aug = transform_aug(image)

image_aug = feature_extractor(image_aug,return_tensors="pt")['pixel_values'][0]
image_aug = image_aug.unsqueeze(0) # [1,3,255,299]
image = feature_extractor(image,return_tensors="pt")['pixel_values'][0]
image = image.unsqueeze(0) # [1,3,255,299]


def create_caption_and_mask(start_token, max_length):
    caption_template = torch.zeros((1, max_length), dtype=torch.long)
    mask_template = torch.ones((1, max_length), dtype=torch.bool)  # 全true

    caption_template[:, 0] = start_token
    mask_template[:, 0] = False  # !!!!!!!!!!!! 第一个false? 为什么要这么做?

    return caption_template, mask_template


caption, cap_mask = create_caption_and_mask(
    start_token, config.max_position_embeddings)
atten_mask = (~cap_mask).long()

metric = nn.MSELoss(reduction='sum')
@torch.no_grad()
def evaluate():
    model.eval()
    for i in range(config.max_position_embeddings - 1):
        """选择不同的方式得到predictions"""
        predictions,img_feature, text_feature =\
             model(image,image_aug, caption, cap_mask, atten_mask)  # [1,128,30522] 这个已经和maxlen 绑定了, 但是自己的代码里面, 里面的pad都去掉了
        print(metric(img_feature,text_feature))    
        print(img_feature)
        # predictions = model.forward_predict_from_text(text_input=caption_input,
        #                             text_input_attn_mask=caption_input_attn_mask,target=caption,target_mask=cap_mask)
        # # 
        """#########################"""
        
        
        
        predictions = predictions[:, i, :]
        if True: print(predictions)
        # print(predictions[:,:20])
        predicted_id = torch.argmax(predictions, axis=-1)
        # print(predictions[:,predicted_id])

        if predicted_id[0] == 102:
            return caption

        caption[:, i+1] = predicted_id[0]
        cap_mask[:, i+1] = False
        atten_mask[:, i+1] = 1

    return caption


output = evaluate()
result = tokenizer.decode(output[0].tolist(), skip_special_tokens=True)
#result = tokenizer.decode(output[0], skip_special_tokens=True)
print(result.capitalize())