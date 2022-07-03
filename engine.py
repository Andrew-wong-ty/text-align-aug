# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch

import math
import sys
import tqdm

from models import utils
from models.caption import Caption


def train_one_epoch(config,model:Caption, criterion_celoss,criterion_closs, data_loader,
                    optimizer, device, epoch, max_norm):
    model.train()
    criterion_celoss.train()
    criterion_closs.train()
    epoch_loss = 0.0
    text_celoss = 0.0
    bert_vit_align_loss = 0.0
    img_aug_align_loss = 0.0
    dual_model_closs = 0.0
    total = len(data_loader)

    with tqdm.tqdm(total=total) as pbar:
        for idx,(images,images_aug, caps, cap_padding_masks,attn_mask) in enumerate(data_loader):

            images = images.to(device)  # 正常的图像
            images_aug = images_aug.to(device)  # 经过了augmentation的图像
            caps = caps.to(device) # tokanized 的caption
            cap_padding_masks = cap_padding_masks.to(device) # shape和caps一样, 用来标记PAD, 有PAD的位置将会变成TRUE, 其余地方是FALSE
            attn_mask = attn_mask.to(device)

            loss = 0.0
            if config.dual_model:
                outputs,img_feature,img_aug_feature,Wrot_gI, text_feature,cls_closs = model(
                    images, images_aug, caps[:,:-1], cap_padding_masks[:,:-1], attn_mask[:,:-1]
                    ) # 向前传播的时候, 使用前n-1个序列
            else:
                outputs,img_feature,img_aug_feature,Wrot_gI, text_feature = model(
                    images, images_aug, caps[:,:-1], cap_padding_masks[:,:-1], attn_mask[:,:-1]
                    ) # 向前传播的时候, 使用前n-1个序列
                cls_closs = 0.0 # dual model 中的closs
            
            
            bert_vit_closs = 0.0
            aug_closs = 0.0
            
            # 计算生成句子的CEloss
            celoss = criterion_celoss(outputs.permute(0, 2, 1), caps[:, 1:])  # outputs.permute(0, 2, 1): [32,30522,128],   [32,128]
            loss+= celoss
            if epoch>=config.pretrain_epochs:
                # 计算BERT和vit的对齐损失
                bert_vit_closs = (
                    criterion_closs(img_feature,text_feature)+
                    criterion_closs(text_feature,img_feature)
                )/2
                loss+=bert_vit_closs

                # 计算BERT和vit的对齐损失
                aug_closs = (
                    criterion_closs(Wrot_gI,img_aug_feature)+
                    criterion_closs(img_aug_feature,Wrot_gI)
                )/2
                loss+=aug_closs

                # dual model中的closs
                if config.dual_model:
                    loss+=cls_closs

            # 记录
            epoch_loss += loss.item()
            text_celoss += celoss.item()
            try:
                bert_vit_align_loss += bert_vit_closs.item()
                img_aug_align_loss += aug_closs.item()
                dual_model_closs += cls_closs.item()
            except:
                pass

            if not math.isfinite(loss.item()):
                print(f'Loss is {loss.item()}, stopping training')
                sys.exit(1)

            info = "[{}/{}] train_loss: {:.4f}  celoss: {:.4f} bert_vit_align_loss: {:.2f} img_aug_align_loss: {:.2f} dual_closs: {:.4f}".format(idx+1,total,
                epoch_loss/(idx+1),
                text_celoss/(idx+1),
                bert_vit_align_loss/(idx+1),
                img_aug_align_loss/(idx+1),
                dual_model_closs/(idx+1),
                )
            pbar.set_description(info)

            optimizer.zero_grad()
            loss.backward()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

            pbar.update(1)

        
            

    return epoch_loss / total

@torch.no_grad()
def evaluate(config,model, criterion_celoss,criterion_closs, data_loader, device):
    model.eval()
    criterion_celoss.eval()

    validation_loss = 0.0
    text_celoss = 0.0
    bert_vit_align_loss = 0.0
    img_aug_align_loss = 0.0
    dual_model_closs = 0.0
    total = len(data_loader)

    with tqdm.tqdm(total=total) as pbar:
        for idx,(images,images_aug, caps, cap_padding_masks,attn_mask) in enumerate(data_loader):
            # samples = utils.NestedTensor(images, masks).to(device)
            images = images.to(device)  # 正常的图像
            images_aug = images_aug.to(device)  # 经过了augmentation的图像
            caps = caps.to(device) # tokanized 的caption
            cap_padding_masks = cap_padding_masks.to(device) # shape和caps一样, 用来标记PAD, 有PAD的位置将会变成TRUE, 其余地方是FALSE
            attn_mask = attn_mask.to(device)

            if config.dual_model:
                outputs,img_feature,img_aug_feature,Wrot_gI, text_feature,cls_closs = model(
                    images, images_aug, caps[:,:-1], cap_padding_masks[:,:-1], attn_mask[:,:-1]
                    ) # 向前传播的时候, 使用前n-1个序列
            else:
                outputs,img_feature,img_aug_feature,Wrot_gI, text_feature = model(
                    images, images_aug, caps[:,:-1], cap_padding_masks[:,:-1], attn_mask[:,:-1]
                    ) # 向前传播的时候, 使用前n-1个序列
                cls_closs = 0.0 # dual model 中的closs
            
            loss = 0.0
            bert_vit_closs = 0.0
            aug_closs = 0.0
            # 计算生成句子的CEloss
            celoss = criterion_celoss(outputs.permute(0, 2, 1), caps[:, 1:])  # outputs.permute(0, 2, 1): [32,30522,128],   [32,128]
            loss+= celoss
            # 计算BERT和vit的对齐损失
            bert_vit_closs = (
                criterion_closs(img_feature,text_feature)+
                criterion_closs(text_feature,img_feature)
            )/2
            loss+=bert_vit_closs

            # 计算BERT和vit的对齐损失
            aug_closs = (
                criterion_closs(Wrot_gI,img_aug_feature)+
                criterion_closs(img_aug_feature,Wrot_gI)
            )/2
            loss+=aug_closs

            # dual model closs
            loss+=cls_closs

            # 记录
            try:
                validation_loss += loss.item()
                text_celoss += celoss.item()
                bert_vit_align_loss += bert_vit_closs.item()
                img_aug_align_loss += aug_closs.item()
                dual_model_closs += cls_closs.item()
            except:
                pass

            pbar.update(1)
    info = "[{}/{}] train_loss: {:.4f}  celoss: {:.4f} bert_vit_align_loss: {:.4f} img_aug_align_loss: {:.4f} dual_closs: {:.2f}".format(idx+1,total,
        validation_loss/(idx+1),
        text_celoss/(idx+1),
        bert_vit_align_loss/(idx+1),
        img_aug_align_loss/(idx+1),
        dual_model_closs/(idx+1),
        )
    print(info)
        
    return validation_loss / total