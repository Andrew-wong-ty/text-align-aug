import torch
from .vit import VisionTransformer, interpolate_pos_embed
import torchvision.transforms as transforms
from sentence_transformers import SentenceTransformer
from functools import partial
import torch.nn as nn
import torch.nn.functional as F



class ContrastiveLoss(nn.Module):
    """
        implementation of ContrastiveLoss 
    """
    def __init__(self,temp:float = 0.07):
        super(ContrastiveLoss, self).__init__()
        self.temp = temp
    def forward(self, Xi, Xt):
        """calculate the contrastive loss of X and Y, which are 3D tensors with shape (bs, max_len, d_model)

            Args:
                Xi: the feature of image
                Xt: the feature of text
            return:
                Contrastive loss of Xi and Xt
        """
        # Xi dot (Xt^T)
        similarity_i2t = Xi @ Xt.transpose(1,2) / self.temp
        # create mask
        mask = torch.zeros_like(similarity_i2t)
        for i in range(len(mask)):mask[i] = mask[i].fill_diagonal_(1)
        # calculate loss
        loss_pre_sample = -torch.sum(F.log_softmax(similarity_i2t,dim=-1)*mask, dim=-1).mean(-1)
        loss = loss_pre_sample.mean()
        return  loss

class ALIGN(nn.Module):
    def __init__(self,args):
        super(ALIGN, self).__init__()
        self.args = args
        bert_model = SentenceTransformer(args.bert_ckpt_path)
        
        self.tokenizer = bert_model[0].tokenizer
        self.sentbert = bert_model[0].auto_model
        self.text_embed_dim = self.sentbert.config.hidden_size
        self.visual_encoder = VisionTransformer(
            img_size=256, patch_size=16, embed_dim=768, depth=12, num_heads=12, 
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))

        if args.load_vision_ckpt:
            checkpoint = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
                map_location="cpu", check_hash=True)
            state_dict = checkpoint["model"]
            pos_embed_reshaped = interpolate_pos_embed(state_dict['pos_embed'], self.visual_encoder)
            state_dict['pos_embed'] = pos_embed_reshaped
            msg = self.visual_encoder.load_state_dict(state_dict,strict=False)
            print(msg)

        ## definition of linear layers
        # W_image is used to scale g(I) to have the same dimension with h(T)
        self.W_image = nn.Linear(state_dict['pos_embed'].shape[1],self.args.max_len) 
        self.W_rot = nn.Linear(self.text_embed_dim,self.text_embed_dim)

        # definition of loss
        self.CLoss = ContrastiveLoss(temp = args.temp)

    def get_text_embeds(self,texts):
        text_output_m= self.tokenizer.batch_encode_plus(texts, 
                                                    max_length=self.args.max_len,  # +2是因为CLS 和SEQ也算进去max_length的
                                                    return_tensors='pt', 
                                                    padding='max_length',
                                                    truncation=True)
        for k,_ in text_output_m.items():
            text_output_m[k] = text_output_m[k].to(self.args.device)
        text_embeds = self.sentbert.forward(**text_output_m)
        text_embeds = text_embeds[0]
        return text_embeds

    @torch.no_grad()  # we do not train the VIT
    def get_img_embeds(self,images):
        return self.visual_encoder(images)

    def forward(self,texts,images,images_aug):
        text_embeds = self.get_text_embeds(texts)  # h(t)
        img_embeds = self.get_img_embeds(images) # g(I), with shape (bs,max-len,d_model)
        img_embeds_aug = self.get_img_embeds(images_aug) # g(I_rot)
        ## tranform the img_embeds to be the same dimension as text_embeds
        # transform g(I)
        img_embeds = img_embeds.permute(0,2,1) 
        img_embeds = self.W_image(img_embeds).permute(0,2,1) # with shape (bs,max-len,d_model)
        # transform g(I_rot)
        img_embeds_aug = img_embeds_aug.permute(0,2,1) 
        img_embeds_aug = self.W_image(img_embeds_aug).permute(0,2,1)
        # normalize them
        text_embeds = F.normalize(text_embeds,-1)
        img_embeds = F.normalize(img_embeds,-1)
        img_embeds_aug = F.normalize(img_embeds_aug,-1)

        ## align h(t) and g(I)
        # calculate CLoss of text_embeds and img_embeds
        closs_text_img = (self.CLoss(text_embeds,img_embeds)
                        +
                        self.CLoss(img_embeds,text_embeds))/2
        
        ## align W_rot*g(I) and g(I_rot)
        trans_img_embeds = self.W_rot(img_embeds)
        closs_aug = (self.CLoss(trans_img_embeds,img_embeds)
                    +
                    self.CLoss(img_embeds,trans_img_embeds))/2

        debug_stop = 1
        
