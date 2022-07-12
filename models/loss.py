import torch
from torch import nn
import torch.nn.functional as F

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