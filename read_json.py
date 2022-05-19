from tqdm import tqdm  
import json
import numpy as np
import os
import random

json_path = "./data/annotations/captions_train2017.json"
img_path = "/home/tywang/myURE/text-align-aug/data/train2017"
with open(json_path,'r') as file:
    data = json.load(file)
annotations = data['annotations']

def get_file_name(img_id,img_id_len=12):
    str_img_id = str(img_id)
    filename = "".join(["0" for i in range(img_id_len-len(str_img_id))])+str_img_id+".jpg"
    return filename
test_jsons = [] # 用于存储图像数据, key有2个 {'image': 图像的路径, 'caption': 图像的caption的text}
lens = []
random.shuffle(annotations)
for item in tqdm(annotations[:50000]):
    file_name = get_file_name(item['image_id'])
    caption = item['caption']
    lens.append(len(caption.split()))
    path = os.path.join(img_path,file_name)
    text_img_pair = {'image':path,'caption':caption}
    test_jsons.append(text_img_pair)
    assert os.path.exists(path)

#保存1000个小样本来作为测试训练数据
with open("/home/tywang/myURE/text-align-aug/data/samples_50000.json","w") as file:
    json.dump(test_jsons,file)
print(np.mean(lens)) # 10

print(max(lens)) # 49
print(np.median(lens))  # maxlength 求32就ok了
debug_stop =1



"""
'image':'/home/tywang/myURE/text-align-aug/data/train2017/000000203564.jpg'
'caption
"""