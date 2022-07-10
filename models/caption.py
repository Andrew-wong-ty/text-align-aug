import torch
from torch import nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from .backbone import build_backbone_vit
from .transformer import build_transformer
from transformers import (
    VisionTextDualEncoderModel,
    VisionTextDualEncoderProcessor,
    ViTFeatureExtractor,
    BertTokenizer,
)


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self,config, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.device = config.device
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, Xi, Xt, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        features = torch.cat([Xi.unsqueeze(1), Xt.unsqueeze(1)], dim=1)
        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)
            features = F.normalize(features,dim = 2)  # 加多一个normalize的操作

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(self.device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(self.device)
        else:
            mask = mask.float().to(self.device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(self.device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


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
        # 这里应该是搞错了, 这里的CL loss是Xi和Xt的某个样本的channel之间做CL, 但其实, 我们需要的是不同的样本之间做CL
        loss_pre_sample = -torch.sum(F.log_softmax(similarity_i2t,dim=-1)*mask, dim=-1).mean(-1)
        loss = loss_pre_sample.mean()
        return  loss


class ContrastiveLossSim(nn.Module):
    """
        implementation of ContrastiveLoss 
    """
    def __init__(self,temp:float = 0.07):
        super(ContrastiveLossSim, self).__init__()
        self.temp = temp
    def forward(self, Xi, Xt):
        """calculate the contrastive loss of X and Y, which are 3D tensors with shape (bs, max_len, d_model)

            Args:
                Xi: the feature of image
                Xt: the feature of text
            return:
                Contrastive loss of Xi and Xt
        """
        # X inner product Y
        inner_product = Xi*Xt/self.temp



        # Xi dot (Xt^T)
        similarity_i2t = Xi @ Xt.transpose(1,2) / self.temp
        # create mask
        mask = torch.zeros_like(similarity_i2t)
        for i in range(len(mask)):mask[i] = mask[i].fill_diagonal_(1)
        # calculate loss
        loss_pre_sample = -torch.sum(F.log_softmax(similarity_i2t,dim=-1)*mask, dim=-1).mean(-1)
        loss = loss_pre_sample.mean()
        return  loss

class MSE(nn.Module):
    """
        implementation of ContrastiveLoss 
    """
    def __init__(self):
        super(MSE, self).__init__()
        self.loss = nn.MSELoss(reduction='sum')
    def forward(self, Xi, Xt):
        """calculate the contrastive loss of X and Y, which are 3D tensors with shape (bs, max_len, d_model)

            Args:
                Xi: the feature of image
                Xt: the feature of text
            return:
                Contrastive loss of Xi and Xt
        """
        loss = self.loss(Xi,Xt)
        return  loss

class Caption(nn.Module):
    def __init__(self,config, backbone, transformer, hidden_dim, vocab_size):
        super().__init__()
        self.config = config
        bert_model = SentenceTransformer(config.pre_train_bert_path)
        self.sentbert = bert_model[0].auto_model  # text encoder 
        self.text_embed_dim = self.sentbert.config.hidden_size  # bert的d_model
        self.backbone = backbone  # image encoder
        self.transformer = transformer # text decoder
        
        self.W_vision = nn.Conv1d(self.config.max_position_embeddings, 197, kernel_size=1)  # 把text_embedding从[bs,n_words,d_model] 投到 [bs,n_channel,d_model]
        
        # self.W_rot = nn.Linear(self.text_embed_dim,self.text_embed_dim)  # 图中的W_rot
        self.W_rot = MLP(hidden_dim, 512, hidden_dim, 2)
        self.mlp = MLP(hidden_dim, 512, vocab_size, 3)  # 投到和vocabulary size一样大小
    
    def get_text_feat(self,full_caption,full_attn_mask):
        """
            用bert得到caption的embedding
        """
        text_embeds = self.sentbert(input_ids = full_caption,attention_mask = full_attn_mask)
        text_embeds = text_embeds[0]
        return text_embeds

    def forward(self, images,images_aug, target, target_mask, attn_mask):


        # 获取图片的特征
        img_feature =self.get_img_feat(images)  # g(I)
        img_aug_feature = self.get_img_feat(images_aug) # g(I_rot)
        Wrot_gI = self.W_rot(img_feature)

        # caption特征
        text_feature = self.W_vision(self.get_text_feat(target,attn_mask))

        # 放进text decoder
        # hs = self.transformer(img_aug_feature, target, target_mask)  # [128,32,256]
        
        if self.config.use_res:
            # 残差连接
            hs = self.transformer(img_feature+text_feature, target, target_mask) 
        else:
            hs = self.transformer(img_feature, target, target_mask)  # [128,32,256]
        out = self.mlp(hs.permute(1, 0, 2)) # hs.permute(1, 0, 2) shape= [32,128,256]
        return out,img_feature,img_aug_feature,Wrot_gI, text_feature

    def get_img_feat(self,images):
        with torch.no_grad():
            img_feature =self.backbone(images).last_hidden_state
        return img_feature


    @torch.no_grad()
    def forward_predict_from_text(self, text_input,text_input_attn_mask , target, target_mask):
        """
            仅在predict的时候使用输入text, 得到这个text的augmentation
        """
        text_feature = self.W_vision(self.get_text_feat(text_input,text_input_attn_mask))

        hs = self.transformer(text_feature, target, target_mask)  # [128,32,256]
        out = self.mlp(hs.permute(1, 0, 2)) # hs.permute(1, 0, 2) shape= [32,128,256]
        return out

    @torch.no_grad()
    def forward_predict_onStep(self, text_input,text_input_attn_mask , target, target_mask,aug=True):
        """
            仅在predict的时候使用输入text, 得到这个text的augmentation
        """
        text_feature = self.W_vision(self.get_text_feat(text_input,text_input_attn_mask))
        if aug:
            text_feature = self.W_rot(text_feature)

        hs = self.transformer(text_feature, target, target_mask)  # [128,32,256]
        out = self.mlp(hs.permute(1, 0, 2)) # hs.permute(1, 0, 2) shape= [32,128,256]
        return out




class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k)
                                    for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class CaptionDual(nn.Module):
    def __init__(self,config,transformer,hidden_dim, vocab_size):
        super().__init__()
        self.config = config
        self.model = VisionTextDualEncoderModel.from_vision_text_pretrained(
            config.backbone, config.pre_train_bert_path
        )
        # 冻结vision_model, visual_projection, text_projection 的参数
        for n, p in self.model.named_parameters():
            if "vision_model" in n:
                p.requires_grad = False
            # if "visual_projection" in n or "text_projection" in n:
            #     p.requires_grad = False
            # print(n," ",p.requires_grad)
        
        self.transformer = transformer
        self.W_vision = nn.Conv1d(self.config.max_position_embeddings, 197, kernel_size=1)  # 把text_embedding从[bs,n_words,d_model] 投到 [bs,n_channel,d_model]
        self.W_rot = MLP(hidden_dim, 512, hidden_dim, 2)
        self.mlp = MLP(hidden_dim, 512, vocab_size, 3)  # 投到和vocabulary size一样大小
    def forward(self, images,images_aug, target, target_mask, attn_mask):
        

        outputs = self.model(
            input_ids=target,
            attention_mask=attn_mask,
            pixel_values=images,
            output_hidden_states =True,
            return_loss=True 
        )
        


        # 获取图片的特征
        img_feature =outputs['vision_model_output'].last_hidden_state  # g(I)
        img_aug_feature = self.model.vision_model(pixel_values=images_aug).last_hidden_state
        Wrot_gI = self.W_rot(img_feature)

        # caption特征
        text_feature = self.W_vision(outputs['text_model_output'].last_hidden_state)
        

        # 放进text decoder
        hs = self.transformer(img_aug_feature, target, target_mask)  # [128,32,256]
        out = self.mlp(hs.permute(1, 0, 2)) # hs.permute(1, 0, 2) shape= [32,128,256]
        return out,img_feature,img_aug_feature,Wrot_gI, text_feature,outputs['loss']

    @torch.no_grad()
    def forward_predict_onStep(self, text_input,text_input_attn_mask , target, target_mask,aug=True):
        """
            仅在predict的时候使用输入text, 得到这个text的augmentation
        """
        text_feature = self.model.text_model(input_ids=text_input,attention_mask=text_input_attn_mask).last_hidden_state
        text_feature = self.W_vision(text_feature)
        if aug:
            text_feature = self.W_rot(text_feature)

        hs = self.transformer(text_feature, target, target_mask)  # [128,32,256]
        out = self.mlp(hs.permute(1, 0, 2)) # hs.permute(1, 0, 2) shape= [32,128,256]
        return out

def build_model(config):
    if not config.dual_model:
        backbone = build_backbone_vit(config)
    transformer = build_transformer(config)
    if config.dual_model:
        model = CaptionDual(config, transformer, config.hidden_dim, config.vocab_size)
    else:
        model = Caption(config, backbone, transformer, config.hidden_dim, config.vocab_size)
    CEloss = torch.nn.CrossEntropyLoss(ignore_index=0)   
    # CLoss = MSE() #contrastive loss
    CLoss = SupConLoss(config)
    # CLoss = ContrastiveLoss(config.temp) #contrastive loss
    return model, CEloss, CLoss