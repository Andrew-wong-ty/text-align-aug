import torch
from torch import Tensor
from .vit import VisionTransformer, interpolate_pos_embed
import torchvision.transforms as transforms
from sentence_transformers import SentenceTransformer
from functools import partial
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(0)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:,:token_embedding.size(1), :])

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
        transformer_decoder_layer = nn.TransformerDecoderLayer(self.text_embed_dim, args.n_head, self.text_embed_dim, 0.1,batch_first=True)
        decoder_norm = nn.LayerNorm(self.text_embed_dim)
        self.transformer_decoder = nn.TransformerDecoder(transformer_decoder_layer, args.num_decoder_layers,norm=decoder_norm)
        self.generate = nn.Linear(self.text_embed_dim,len(self.tokenizer)) # project embedding_dim to vocabulary size

        # definition of loss
        self.CLoss = ContrastiveLoss(temp = args.temp)
        self.CEloss = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)

        # others
        self.position_encoder = PositionalEncoding(self.text_embed_dim,0.1)


    def tokenize(self,texts,mode):
        """Tokenize a batch of texts
            Args:
                text: List[str]
                mode: choice in `['all','first','last']`
                      'all': tokenize the whole sentence,
                      'first': tokenize the first n-1 words in the sentence
                      'last': tokenize the last n-1 words in a sentence

        """
        assert mode in ['all','first','last']
        text_tokenization= self.tokenizer.batch_encode_plus(texts, 
                                                    max_length=self.args.max_len,  # +2是因为CLS 和SEQ也算进去max_length的
                                                    return_tensors='pt', 
                                                    padding='max_length',
                                                    truncation=True)  
        for k,_ in text_tokenization.items():
            if mode == 'first':
                text_tokenization[k] = text_tokenization[k][:,:-1].to(self.args.device)
            elif mode == 'last':
                text_tokenization[k] = text_tokenization[k][:,1:].to(self.args.device)
            else:
                text_tokenization[k] = text_tokenization[k].to(self.args.device)
        return text_tokenization
        
    def get_text_embeds(self,text_tokenization):
        """
            input_ids: [CLS] xxx [SEP] [PAD]..[PAD]
            token_type_ids: 全0
            attention_mask: 1..10..0 [PAD]的地方就是0
        """
        text_embeds = self.sentbert.forward(**text_tokenization)
        text_embeds = text_embeds[0]
        return text_embeds

    @torch.no_grad()  # we do not train the VIT
    def get_img_embeds(self,images):
        return self.visual_encoder(images)

    def forward(self,texts,images,images_aug):
        text_embedding = self.get_text_embeds(self.tokenize(texts,mode='all'))  # h(t)
        img_embedding = self.get_img_embeds(images) # g(I), with shape (bs,max-len,d_model)
        img_embedding_aug = self.get_img_embeds(images_aug) # g(I_rot)
        # normalize them
        text_embeds = F.normalize(text_embedding,dim=-1)
        img_embeds = F.normalize(img_embedding,dim=-1)
        img_embeds_aug = F.normalize(img_embedding_aug,dim=-1)
        ## tranform the img_embeds to be the same dimension as text_embeds
        # transform g(I)
        img_embeds = img_embeds.permute(0,2,1) 
        img_embeds = self.W_image(img_embeds).permute(0,2,1) # with shape (bs,max-len,d_model)
        # transform g(I_rot)
        img_embeds_aug = img_embeds_aug.permute(0,2,1) 
        img_embeds_aug = self.W_image(img_embeds_aug).permute(0,2,1)
        

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

        ## decode
        tgt_input = self.tokenize(texts,mode='first')
        tgt_output = self.tokenize(texts,mode='last').data['input_ids']
        
        tgt = self.get_text_embeds(tgt_input) # the embedding of the first n-1 words
        tgt = self.position_encoder.forward(tgt)

        tgt_mask, tgt_padding_mask = create_mask(self.args,tgt_input.data['input_ids'],self.tokenizer.pad_token_id)

        output = self.transformer_decoder.forward(
            tgt=tgt,
            memory=trans_img_embeds,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_padding_mask
        )
        logits = self.generate(output)
        celoss = self.CEloss(logits.reshape(-1, logits.shape[-1]), tgt_output.reshape(-1)) 

        return closs_text_img,closs_aug,celoss




def generate_square_subsequent_mask(args,sz):
    mask = (torch.triu(torch.ones((sz, sz), device=args.device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(args,tgt,PAD_IDX):
    # src_seq_len = src.shape[1]
    tgt_seq_len = tgt.shape[1]

    tgt_mask = generate_square_subsequent_mask(args,tgt_seq_len)
    # src_mask = torch.zeros((src_seq_len, src_seq_len),device=args.device).type(torch.bool)

    tgt_padding_mask = (tgt == PAD_IDX)
    return tgt_mask, tgt_padding_mask

