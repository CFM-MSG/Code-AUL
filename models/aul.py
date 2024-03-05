import os
import json
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn import init
from timm.models.layers import trunc_normal_
from functools import partial
import numpy as np

from models.swin_transformer import SwinTransformer, interpolate_relative_pos_embed
from models.bert import BertConfig, BertForMaskedLM, BertModel
from utils import read_json

from models import mim_blocks


class CrossEntropyLabelSmooth(nn.Module):
    def __init__(self, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        _, num_classes = inputs.shape
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss


class AllGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, rank, world_size):
        output = [torch.empty_like(tensor) for _ in range(world_size)]
        dist.all_gather(output, tensor)
        ctx.rank = rank
        ctx.batch_size = tensor.shape[0]
        return torch.cat(output, 0)

    @staticmethod
    def backward(ctx, grad_output):
        return (
            grad_output[ctx.batch_size * ctx.rank: ctx.batch_size * (ctx.rank + 1)],
            None,
            None
        )


allgather = AllGather.apply


def build_vision_encoder(config, load_params=False):
    print('use_swin')
    vision_config = read_json(config['vision_config'])
    assert config['image_res'] == vision_config['image_res']
    assert config['patch_size'] == 32
    vision_width = vision_config['vision_width']

    vision_encoder = SwinTransformer(img_size=vision_config['image_res'],
                                     h=vision_config['h'],
                                     w=vision_config['w'],
                                     patch_size=4,
                                     in_chans=3,
                                     embed_dim=vision_config['embed_dim'],
                                     depths=vision_config['depths'],
                                     num_heads=vision_config['num_heads'],
                                     window_size=vision_config['window_size'],
                                     mlp_ratio=4.,
                                     qkv_bias=True,
                                     drop_rate=0.0,
                                     drop_path_rate=0.1,
                                     ape=False,
                                     patch_norm=True,
                                     use_checkpoint=False)

    if load_params:
        state_dict = torch.load(vision_config['ckpt'], map_location="cpu")['model']
        window_size = vision_config['window_size']

        for k in list(state_dict.keys()):
            if 'relative_position_bias_table' in k:
                if 'layers.3' in k:
                    window_size = 4
                dst_num_pos = (2 * window_size - 1) ** 2
                state_dict[k] = interpolate_relative_pos_embed(state_dict[k], dst_num_pos, param_name=k)
            elif ('relative_position_index' in k) or ('attn_mask' in k):
                del state_dict[k]
        print("###  build_vision_encoder: ", flush=True)
        msg = vision_encoder.load_state_dict(state_dict, strict=False)
        print("missing_keys: ", msg.missing_keys)
        print("unexpected_keys: ", msg.unexpected_keys)

    return vision_encoder, vision_width


def build_text_encoder(config, vision_width, load_text_params=False, use_mlm_loss=False, config_text=None):
    init_params = []

    if config_text is None:
        config_text = BertConfig.from_json_file(config['text_config'])

    config_text.encoder_width = vision_width

    if use_mlm_loss:
        text_encoder, msg = BertForMaskedLM.from_pretrained(config['text_encoder'], config=config_text,
                                                            output_loading_info=True)
        if ('init_cross' in config.keys()) and config['init_cross']:
            init_params.extend(['text_encoder.' + n for n in msg['missing_keys']])
            print("###  init_params.extend --> cross attention  ###")

        if load_text_params:
            print("###  build_text_encoder --> Load BERT: ")
            for k, v in msg.items():
                print(f"{k}: {sorted(v)}")
    return text_encoder, init_params


def build_mlp(input_dim, output_dim):
    return nn.Sequential(
        nn.Linear(input_dim, input_dim * 2),
        nn.LayerNorm(input_dim * 2),
        nn.GELU(),
        nn.Linear(input_dim * 2, output_dim)
    )


def attr_mlp(input_dim, inter_dim, output_dim, after_cross, dropout_p):
    if after_cross:
        new_mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, inter_dim),
            nn.LayerNorm(inter_dim),
            nn.Dropout(p=dropout_p),
            nn.Linear(inter_dim, output_dim)
        )
    else:
        new_mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, inter_dim),
            nn.BatchNorm1d(inter_dim),
            nn.Dropout(p=dropout_p),
            nn.Linear(inter_dim, output_dim)
        )
    init.normal_(new_mlp[1].weight.data, std=0.00001)
    init.constant_(new_mlp[1].bias.data, 0.0)
    init.normal_(new_mlp[4].weight.data, std=0.00001)
    init.constant_(new_mlp[4].bias.data, 0.0)
    return new_mlp


def load_pretrained(ckpt_rpath, config, is_eval=False, load_text=False):
    checkpoint = torch.load(ckpt_rpath, map_location='cpu')
    state_dict = checkpoint['model'] if 'model' in checkpoint.keys() else checkpoint
    if is_eval:
        return state_dict

    print("### Loading pretrained vision encoder", flush=True)

    if load_text:
        print("### Loading pretrained text encoder", flush=True)
        for key in list(state_dict.keys()):
            if 'text_encoder.' in key:
                if not config['mlm']:
                    if 'bert.' in key:
                        encoder_key = key.replace('bert.', '')
                        state_dict[encoder_key] = state_dict[key]
                        del state_dict[key]
                else:
                    if 'bert.' not in key and 'cls' not in key:
                        encoder_key = key.replace('text_encoder.', 'text_encoder.bert.')
                        state_dict[encoder_key] = state_dict[key]
                        del state_dict[key]

    return state_dict


class AUL(nn.Module):
    def __init__(self, config=None, load_vision_params=False, load_text_params=False,
                 use_contrastive_loss=False, use_matching_loss=False,
                 use_mlm_loss=False, config_text=None):
        super().__init__()
        self.init_params = []

        self.vision_encoder, vision_width = build_vision_encoder(config, load_params=load_vision_params)
        self.vision_width = vision_width
        if config['mim']:
            self.mask_token = nn.Parameter(torch.zeros([1, 3, 32, 32]))
            self.mim_gen = mim_blocks.mim_decoder(config)
        if config['id']:
            self.classfier = nn.Linear(config['embed_dim'], 11003)

        else:
            # text & cross-modal
            self.text_encoder, init_params = build_text_encoder(config, vision_width=vision_width,
                                                                load_text_params=load_text_params,
                                                                use_mlm_loss=use_mlm_loss, config_text=config_text)
            self.text_width = self.text_encoder.config.hidden_size 
            self.init_params.extend(init_params)
            if 0 < config['LabelSmooth'] < 1:
                self.new_cross_entropy = CrossEntropyLabelSmooth(epsilon=config['LabelSmooth'])
                self.add_label_smooth = True
            else:
                self.add_label_smooth = False
            if config['lr_2']:
                for i in range(2, 4):
                    for name, param in self.vision_encoder.layers[i].named_parameters():
                        self.init_params.extend(['vision_encoder.layers.' + str(i) + '.' + name])
                if config['mlm']:
                    self.init_params.extend(
                        ['text_encoder.cls.' + n for n, _ in self.text_encoder.cls.named_parameters()])
                    temp_name = 'text_encoder.bert.encoder.layer.'
                    temp_encoder = self.text_encoder.bert
                else:
                    temp_name = 'text_encoder.encoder.layer.'
                    temp_encoder = self.text_encoder
                temp_list = [4, 5, 10, 11]
                for i in temp_list:
                    for name, param in temp_encoder.encoder.layer[i].named_parameters():
                        self.init_params.extend([temp_name + str(i) + '.' + name])

            if use_contrastive_loss:
                self.embed_dim = config['embed_dim']
                self.vision_proj = nn.Linear(self.vision_width, self.embed_dim)
                self.text_proj = nn.Linear(self.text_width, self.embed_dim)
                self.temp = nn.Parameter(torch.ones([]) * config['temp'])
                if config['lr_2']:
                    self.init_params.extend(['vision_proj.' + n for n, _ in self.vision_proj.named_parameters()])
                    self.init_params.extend(['text_proj.' + n for n, _ in self.text_proj.named_parameters()])

            if use_matching_loss:
                self.itm_head = build_mlp(input_dim=self.text_width, output_dim=2)
                if config['lr_2']:
                    self.init_params.extend(['itm_head.' + n for n, _ in self.itm_head.named_parameters()])

    def load_pretrained(self, ckpt_rpath, config, is_eval=False):
        state_dict = load_pretrained(ckpt_rpath, config, is_eval=is_eval, load_text=True)
        msg = self.load_state_dict(state_dict, strict=False)
        print('load checkpoint from %s' % ckpt_rpath)
        print("missing_keys: ", [p for p in msg.missing_keys if 'vision_encoder' not in p])
        print("unexpected_keys: ", msg.unexpected_keys)

    def get_vision_embeds(self, image):
        image_embeds = self.vision_encoder(image)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
        return image_embeds, image_atts

    def get_text_embeds(self, text_ids, text_atts):
        encoder = self.text_encoder.bert if hasattr(self.text_encoder, 'bert') else self.text_encoder
        return encoder(text_ids, attention_mask=text_atts, return_dict=True, mode='text').last_hidden_state

    def get_cross_embeds(self, image_embeds, image_atts, text_ids=None, text_embeds=None, text_atts=None):
        assert text_atts is not None
        encoder = self.text_encoder.bert if hasattr(self.text_encoder, 'bert') else self.text_encoder
        if text_embeds is not None:
            return encoder(encoder_embeds=text_embeds,
                           attention_mask=text_atts,
                           encoder_hidden_states=image_embeds,
                           encoder_attention_mask=image_atts,
                           return_dict=True,
                           mode='fusion',
                           ).last_hidden_state
        elif text_ids is not None:
            return encoder(text_ids,
                           attention_mask=text_atts,
                           encoder_hidden_states=image_embeds,
                           encoder_attention_mask=image_atts,
                           return_dict=True,
                           ).last_hidden_state
        else:
            raise ValueError

    def get_features(self, image_embeds=None, text_embeds=None):
        if image_embeds is None:
            text_feat = self.text_proj(text_embeds[:, 0, :])
            return text_feat
        elif text_embeds is None:
            image_feat = self.vision_proj(image_embeds[:, 0, :])
            return image_feat
        else:
            image_feat = self.vision_proj(image_embeds[:, 0, :])
            text_feat = self.text_proj(text_embeds[:, 0, :])
            return image_feat, text_feat

    def get_t2i(self, image_feat, text_feat, idx=None):
        assert image_feat.size(-1) == self.embed_dim
        assert text_feat.size(-1) == self.embed_dim
        image_feat = F.normalize(image_feat, dim=-1)
        text_feat = F.normalize(text_feat, dim=-1)

        image_feat_all = allgather(image_feat, torch.distributed.get_rank(), torch.distributed.get_world_size())
        text_feat_all = allgather(text_feat, torch.distributed.get_rank(), torch.distributed.get_world_size())
        logits = image_feat_all @ text_feat_all.t() / self.temp
        bsz = image_feat_all.shape[0]


        if idx is None:
            labels = torch.arange(bsz, device=image_feat.device)
            loss_t2i = F.cross_entropy(logits.t(), labels)
            return loss_t2i
        else:
            idx = idx.view(-1, 1)
            assert idx.size(0) == image_feat.size(0)
            idx_all = allgather(idx, torch.distributed.get_rank(), torch.distributed.get_world_size())
            pos_idx = torch.eq(idx_all, idx_all.t()).float()
            labels = pos_idx / pos_idx.sum(1, keepdim=True)

            loss_t2i = -torch.sum(F.log_softmax(logits.t(), dim=1) * labels, dim=1).mean()
            return loss_t2i

    def get_lu_loss(self, image_features, text_features, augmented_features, total_epoch, cur_epoch):
        se = self.get_t2i(augmented_features, text_features)
        std = torch.std(augmented_features)
        inv_std = torch.exp(-std)

        mse = torch.mean(inv_std * se)
        reg = torch.mean(std)
        L_u = (mse + reg) / 2
        L_info = self.get_t2i(image_features, text_features)
        gamma = np.exp(- cur_epoch / total_epoch)
        return gamma * L_u + (1 - gamma) * L_info

    def add_gaussian_noisy(self, img, alpha_scale=1, beta_scale=1):
        x = img.unsqueeze(dim=1)
        std, mean = torch.std_mean(x, dim=0)
        normal_alpha = torch.distributions.Normal(loc=1, scale=std)
        normal_beta = torch.distributions.Normal(loc=mean, scale=std)
        alpha = alpha_scale * normal_alpha.sample([x.shape[0]])
        beta = beta_scale * normal_beta.sample([x.shape[0]])
        x = nn.functional.instance_norm(x)
        x = alpha * x + beta
        return x.squeeze(dim=1)

    def build_masks_for_one_batch(self, batch_size, mask_ratio=0.75, patch_num=48):
        mask_length = int(patch_num * mask_ratio)
        mask_batch = []
        for i in range(int(batch_size)):
            mask_idx = torch.randperm(patch_num)[:mask_length]
            mask1 = torch.zeros([patch_num])
            mask1[mask_idx] = 1
            mask_batch.append(mask1)
        mask = torch.stack(mask_batch, dim=0)
        return mask

    def build_masked_image(self, image, masks):
        assert masks is not None
        image = image.cuda()
        masks = masks.cuda()
        B, C, H, W = image.shape
        mask_tokens = self.mask_token.repeat(B, 1, 12, 4)
        temp_mask = masks.reshape(B, 12, 4).unsqueeze(1).repeat(1, 3, 32, 32)
        x = torch.mul(image, (1.0 - temp_mask)) + torch.mul(temp_mask, mask_tokens)
        return x
    
    def get_unmasked_image(self, image, masks):
        image = image.cuda()
        masks = masks.cuda()
        B = image.shape[0]
        temp_mask = masks.reshape(B, 12, 4).unsqueeze(1).repeat(1, 3, 32, 32)
        reserve_token = (temp_mask == 1)
        image = image[reserve_token]
        return image


    def KL(self, alpha, c):
        beta = torch.ones((1, c)).to(alpha.device)
        S_alpha = torch.sum(alpha, dim=1, keepdim=True)
        S_beta = torch.sum(beta, dim=1, keepdim=True)
        lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
        lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
        dg0 = torch.digamma(S_alpha)
        dg1 = torch.digamma(alpha)
        kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
        return kl

    def mse_loss(self, label, alpha, c, lambda2):
        S = torch.sum(alpha, dim=1, keepdim=True)
        E = alpha - 1
        m = alpha / S
        A = torch.sum((label - m) ** 2, dim=1, keepdim=True)
        B = torch.sum(alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True)
        alp = E * (1 - label) + 1
        C = lambda2 * self.KL(alp, c)
        return (A + B) + C

    def mse_loss_tanh(self, label, alpha, c, lambda2):
        S = torch.sum(alpha, dim=1, keepdim=True)
        E = alpha - 1
        m = alpha / S
        u_1 = alpha.size(0) / S
        min_u = torch.min(u_1)
        max_u = torch.max(u_1)
        u_norm = (u_1 - min_u) / (max_u - min_u)
        A = torch.sum((label - m) ** 2, dim=1, keepdim=True)
        B = torch.sum(alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True)
        alp = E * (1 - label) + 1
        C = lambda2 * self.KL(alp, c)
        return (A + B) + C
    
    def get_alpha_t(self, sims):
        evidences = torch.exp(torch.tanh(sims)/0.1)
        sum_e = evidences + evidences.t()
        norm_e = sum_e / torch.sum(sum_e, dim=1, keepdim=True)
        alpha_i2t = evidences + 1
        alpha_t2i = evidences.t() + 1
        return alpha_i2t, alpha_t2i, norm_e

    def get_alpha(self,sims):
        evidences = F.relu(sims)
        sum_e = evidences + evidences.t()
        norm_e = sum_e / torch.sum(sum_e, dim=1, keepdim=True)
        alpha_i2t = evidences + 1
        alpha_t2i = evidences.t() + 1
        return alpha_i2t, alpha_t2i, norm_e

    def course_function(self, epoch, total_epoch, total_snippet_num, amplitude):
        idx = torch.arange(total_snippet_num)
        theta = 2 * (idx + 0.5) / total_snippet_num - 1
        delta = - 2 * epoch / total_epoch + 1
        curve = amplitude * torch.tanh(theta * delta) + 1

        return curve

    def get_unc_loss(self, image_fetures, text_fetures, pid, epoch, total_epoch, amplititude=0.7):
        """
        Similarity Distribution Matching
        """
        batch_size = image_fetures.shape[0]
        if pid == None:
            pid = torch.arange(batch_size, device=image_fetures.device)
        pid = pid.reshape((batch_size, 1)) 
        pid_dist = pid - pid.t()
        labels = (pid_dist == 0).float()

        image_norm = image_fetures / image_fetures.norm(dim=1, keepdim=True)
        text_norm = text_fetures / text_fetures.norm(dim=1, keepdim=True)

        t2i_cosine_theta = text_norm @ image_norm.t()
        i2t_cosine_theta = t2i_cosine_theta.t()

        text_proj_image = t2i_cosine_theta

        labels_distribute = labels / labels.sum(dim=1)
        alpha_t2i, alpha_i2t, _ = self.get_alpha(text_proj_image)
        alpha_t2i_t, alpha_i2t_t, _ = self.get_alpha_t(text_proj_image)
        u_1 = batch_size / torch.sum((0.5 * alpha_i2t_t + 0.5 * alpha_i2t) / 1.0, dim=1, keepdim=True)
        
        total_u_num = u_1.shape[0]
        
        curve = self.course_function(epoch, total_epoch, total_u_num, amplititude)
        curve = curve.cuda()
        loss = (self.mse_loss_tanh(labels_distribute, alpha_t2i_t, batch_size, 0.1)) + 1.0 * (self.mse_loss_tanh(labels_distribute, alpha_i2t_t, batch_size, 0.1))
        loss1 = (self.mse_loss(labels, alpha_t2i, batch_size, 0.1)) + 1.0 * (self.mse_loss(labels, alpha_i2t, batch_size, 0.1))
        total_loss_guide = 0.5 * loss + 0.5 * loss1

        _, uct_indices = torch.sort(u_1, dim=0)
        sorted_curve = torch.gather(curve.repeat(128, 1), 0, uct_indices).cuda()
        uct_guide_loss = (torch.mul(sorted_curve, total_loss_guide).mean()).cuda()
        
        return uct_guide_loss

    def get_id_loss(self, image_logits, text_logits, labels):
        criterion = nn.CrossEntropyLoss(reduction="mean")

        loss = criterion(image_logits, labels) + criterion(text_logits, labels)
        
        return loss / 2

    def get_sdm_loss(self, image_fetures, text_fetures, pid, logit_scale, image_id=None, factor=0.3, epsilon=1e-8):
        batch_size = image_fetures.shape[0]
        pid = pid.reshape((batch_size, 1)) 
        pid_dist = pid - pid.t()
        labels = (pid_dist == 0).float()

        if image_id != None:

            image_id = image_id.reshape((-1, 1))
            image_id_dist = image_id - image_id.t()
            image_id_mask = (image_id_dist == 0).float()
            labels = (labels - image_id_mask) * factor + image_id_mask


        image_norm = image_fetures / image_fetures.norm(dim=1, keepdim=True)
        text_norm = text_fetures / text_fetures.norm(dim=1, keepdim=True)

        t2i_cosine_theta = text_norm @ image_norm.t()
        i2t_cosine_theta = t2i_cosine_theta.t()

        text_proj_image = t2i_cosine_theta / logit_scale
        image_proj_text = i2t_cosine_theta / logit_scale


        labels_distribute = labels / labels.sum(dim=1)

        i2t_pred = F.softmax(image_proj_text, dim=1)
        i2t_loss = i2t_pred * (F.log_softmax(image_proj_text, dim=1) - torch.log(labels_distribute + epsilon))
        t2i_pred = F.softmax(text_proj_image, dim=1)
        t2i_loss = t2i_pred * (F.log_softmax(text_proj_image, dim=1) - torch.log(labels_distribute + epsilon))

        loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))

        return loss

    def get_mim_loss(self, recon, img):
        l1 = nn.L1Loss()
        return l1(recon,img)

    def get_contrastive_loss(self, image_feat, text_feat, idx=None):
        assert image_feat.size(-1) == self.embed_dim
        assert text_feat.size(-1) == self.embed_dim
        image_feat = F.normalize(image_feat, dim=-1)
        text_feat = F.normalize(text_feat, dim=-1)

        image_feat_all = allgather(image_feat, torch.distributed.get_rank(), torch.distributed.get_world_size())
        text_feat_all = allgather(text_feat, torch.distributed.get_rank(), torch.distributed.get_world_size())
        logits = image_feat_all @ text_feat_all.t() / self.temp
        bsz = image_feat_all.shape[0]

        if idx is None:
            labels = torch.arange(bsz, device=image_feat.device)
            loss_i2t = F.cross_entropy(logits, labels)
            loss_t2i = F.cross_entropy(logits.t(), labels)
            return (loss_i2t + loss_t2i) / 2
        else:
            idx = idx.view(-1, 1)
            assert idx.size(0) == image_feat.size(0)
            idx_all = allgather(idx, torch.distributed.get_rank(), torch.distributed.get_world_size())
            pos_idx = torch.eq(idx_all, idx_all.t()).float()
            labels = pos_idx / pos_idx.sum(1, keepdim=True)

            loss_i2t = -torch.sum(F.log_softmax(logits, dim=1) * labels, dim=1).mean()
            loss_t2i = -torch.sum(F.log_softmax(logits.t(), dim=1) * labels, dim=1).mean()
            return (loss_i2t + loss_t2i) / 2

    def get_matching_loss(self, image_embeds, image_atts, image_feat, text_embeds, text_atts, text_feat, idx=None):
        bs = image_embeds.size(0)

        image_feat = F.normalize(image_feat, dim=-1)
        text_feat = F.normalize(text_feat, dim=-1)

        with torch.no_grad():
            sim_i2t = image_feat @ text_feat.t() / self.temp
            sim_t2i = text_feat @ image_feat.t() / self.temp

            weights_i2t = F.softmax(sim_i2t, dim=1) + 1e-5
            weights_t2i = F.softmax(sim_t2i, dim=1) + 1e-5

            if idx is None:
                weights_i2t.fill_diagonal_(0)
                weights_t2i.fill_diagonal_(0)
            else:
                idx = idx.view(-1, 1)
                assert idx.size(0) == bs
                mask = torch.eq(idx, idx.t())
                weights_i2t.masked_fill_(mask, 0)
                weights_t2i.masked_fill_(mask, 0)

        image_embeds_neg = []
        image_atts_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2i[b], 1).item()
            image_embeds_neg.append(image_embeds[neg_idx])
            image_atts_neg.append(image_atts[neg_idx])
        image_embeds_neg = torch.stack(image_embeds_neg, dim=0)
        image_atts_neg = torch.stack(image_atts_neg, dim=0)

        text_embeds_neg = []
        text_atts_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            text_embeds_neg.append(text_embeds[neg_idx])
            text_atts_neg.append(text_atts[neg_idx])
        text_embeds_neg = torch.stack(text_embeds_neg, dim=0)
        text_atts_neg = torch.stack(text_atts_neg, dim=0)

        text_embeds_all = torch.cat([text_embeds, text_embeds_neg], dim=0)
        text_atts_all = torch.cat([text_atts, text_atts_neg], dim=0)
        image_embeds_all = torch.cat([image_embeds_neg, image_embeds], dim=0)
        image_atts_all = torch.cat([image_atts_neg, image_atts], dim=0)

        cross_pos = self.get_cross_embeds(image_embeds, image_atts, text_embeds=text_embeds,
                                          text_atts=text_atts)[:, 0, :]
        cross_neg = self.get_cross_embeds(image_embeds_all, image_atts_all, text_embeds=text_embeds_all,
                                          text_atts=text_atts_all)[:, 0, :]

        output = self.itm_head(torch.cat([cross_pos, cross_neg], dim=0))
        itm_labels = torch.cat([torch.ones(bs, dtype=torch.long),
                                torch.zeros(2 * bs, dtype=torch.long)], dim=0).to(image_embeds.device)
        itm_loss = F.cross_entropy(output, itm_labels)

        return itm_loss

    def get_mlm_loss(self, text_ids_masked, text_atts, image_embeds, image_atts, masked_pos, masked_ids):
        return self.text_encoder(text_ids_masked,
                                 attention_mask=text_atts,
                                 encoder_hidden_states=image_embeds,
                                 encoder_attention_mask=image_atts,
                                 return_dict=True,
                                 labels=masked_ids,
                                 masked_pos=masked_pos).loss

    def label_smooth_loss(self, inputs, targets):
        bs = inputs.size(0)
        inputs_neg = []
        targets_neg = []
        for b in range(bs):
            if targets[b] != -1:
                inputs_neg.append(inputs[b])
                targets_neg.append(targets[b])
        if not inputs_neg:
            return F.cross_entropy(inputs, targets, ignore_index=-1)
        inputs = torch.stack(inputs_neg, dim=0)
        targets = torch.stack(targets_neg, dim=0)
        return self.new_cross_entropy(inputs, targets)

    def get_contrastive_loss_attr(self, image_feat, text_feat, label):
        image_feat = F.normalize(image_feat, dim=-1)
        text_feat = F.normalize(text_feat, dim=-1)
        logits = image_feat @ text_feat.t() / self.temp
        l = 0
        for i in range(label.size(1)):
            left = 2 * i
            right = 2 * i + 2
            if self.add_label_smooth:
                l = l + self.label_smooth_loss(logits[:, left:right], label[:, i])
            else:
                l = l + F.cross_entropy(logits[:, left:right], label[:, i], ignore_index=-1)

        return l / label.size(1)

    def get_matching_loss_attr(self, image_embeds, image_atts, text_embeds, text_atts, label):
        bs = image_embeds.size(0)

        labels = []
        for i in range(label.size(1)):
            l = 1 - label[:, i]
            l = torch.where(l == 2, -1, l)
            labels.append(l)
            labels.append(label[:, i])
        labels = torch.stack(labels, dim=1)

        r = random.sample(range(0, text_embeds.size(0)), 5)
        ll = 0
        for t in r:
            text_embeds_0 = text_embeds[t].repeat(bs, 1, 1)
            text_atts_0 = text_atts[t].repeat(bs, 1, 1)
            cross_0 = self.get_cross_embeds(image_embeds, image_atts, text_embeds=text_embeds_0,
                                            text_atts=text_atts_0)[:, 0, :]
            output_0 = self.itm_head(cross_0)
            if self.add_label_smooth:
                ll = ll + self.label_smooth_loss(output_0, labels[:, t])
            else:
                ll = ll + F.cross_entropy(output_0, labels[:, t], ignore_index=-1)
        return ll / 5

    def get_mlm_loss_attr(self, text_ids_masked, text_atts, image_embeds, image_atts, masked_pos, masked_ids, label):

        labels = []
        for i in range(label.size(1)):
            l = 1 - label[:, i]
            l = torch.where(l == 2, -1, l)
            labels.append(l)
            labels.append(label[:, i])
        labels = torch.stack(labels, dim=1)

        image_embeds_pos = []
        image_atts_pos = []
        text_ids_masked_pos = []
        text_atts_pos = []
        masked_pos_pos = []
        masked_ids_pos = []
        for b in range(text_atts.size(0)):
            temp_label = labels[:, b]
            temp_label = torch.where(temp_label == -1, 0, temp_label)
            if torch.count_nonzero(temp_label).item() > 0:
                text_ids_masked_pos.append(text_ids_masked[b])
                text_atts_pos.append(text_atts[b])
                masked_pos_pos.append(masked_pos[b])
                masked_ids_pos.append(masked_ids[b])
                idx = torch.multinomial(temp_label.float(), 1).item()
                image_embeds_pos.append(image_embeds[idx])
                image_atts_pos.append(image_atts[idx])

        image_embeds_pos = torch.stack(image_embeds_pos, dim=0)
        image_atts_pos = torch.stack(image_atts_pos, dim=0)
        text_ids_masked_pos = torch.stack(text_ids_masked_pos, dim=0)
        text_atts_pos = torch.stack(text_atts_pos, dim=0)
        masked_pos_pos = torch.stack(masked_pos_pos, dim=0)
        masked_ids_pos = torch.stack(masked_ids_pos, dim=0)

        loss = self.text_encoder(text_ids_masked_pos,
                                 attention_mask=text_atts_pos,
                                 encoder_hidden_states=image_embeds_pos,
                                 encoder_attention_mask=image_atts_pos,
                                 return_dict=True,
                                 labels=masked_ids_pos,
                                 masked_pos=masked_pos_pos).loss
        return loss

    def get_match_loss(self, img_cls, text_cls):
        batch_size = img_cls.shape[0]
        pos_score = torch.cosine_similarity(img_cls, text_cls, dim=-1)

        neg_idx = torch.randperm(int(batch_size))[:batch_size]
        neg_img_cls = img_cls[neg_idx]
        neg_text_cls  = text_cls[neg_idx]
        neg_score = torch.cosine_similarity(neg_img_cls, neg_text_cls, dim=-1)
        scores = torch.stack([pos_score, neg_score], 0)


        label = torch.tensor([1, 0]).to(img_cls.device)
        return F.cross_entropy(scores, label)