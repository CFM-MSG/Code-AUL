import torch
from models import AUL, load_pretrained, AllGather
import torch.nn as nn
import torch.nn.functional as F


class AUL_Retrieval(AUL):
    def __init__(self, config):
        super().__init__(config, load_vision_params=config['load_params'], load_text_params=config['load_params'],
                         use_contrastive_loss=True, use_matching_loss=True, use_mlm_loss=config['mlm'])

        self.mlm = config['mlm']
        self.total_epoch = config['schedular']['epochs']

        self.lu = config['lu']
        self.unc = config['unc']
        self.mim = config['mim']
        self.sdm = config['sdm']
        self.id = config['id']
        self.match = config['match']
        self.apply_mlm = config['apply_mlm']
        self.eda = config['eda']
        if ('attr' in config.keys()) and config['attr']:
            self.attr = True
        else:
            self.attr = False

    def load_pretrained(self, ckpt_rpath, config, is_eval=False):
        state_dict = load_pretrained(ckpt_rpath, config, is_eval=is_eval, load_text=True)
        msg = self.load_state_dict(state_dict, strict=False)
        print('load checkpoint from %s' % ckpt_rpath)
        print("missing_keys: ", [p for p in msg.missing_keys if 'vision_encoder' not in p])
        print("vision_encoder missing_keys: ", [p for p in msg.missing_keys if 'vision_encoder' in p])
        print("unexpected_keys: ", msg.unexpected_keys)

    def forward(self, image, text_ids, text_atts, text_ids_masked=None, masked_pos=None, masked_ids=None,
                idx=None, attr_text_ids=None, attr_text_atts=None, attr_text_ids_masked=None,
                attr_masked_pos=None, attr_masked_ids=None, label=None, text_ids_eda=None, text_atts_eda=None, cur_epoch=None):
        

        if self.attr:
            image_embeds, image_atts = self.get_vision_embeds(image)
            text_embeds = self.get_text_embeds(text_ids, text_atts)
            image_feat, text_feat = self.get_features(image_embeds, text_embeds)

            attr_text_embeds = self.get_text_embeds(attr_text_ids, attr_text_atts)
            attr_text_feat = self.get_features(text_embeds=attr_text_embeds)

            attr_loss_itc = self.get_contrastive_loss_attr(image_feat, attr_text_feat, label)
            attr_loss_itm = self.get_matching_loss_attr(image_embeds, image_atts, attr_text_embeds, attr_text_atts,
                                                        label)

            loss_itc = self.get_contrastive_loss(image_feat, text_feat, idx=idx)
            loss_itm = self.get_matching_loss(image_embeds, image_atts, image_feat,
                                              text_embeds, text_atts, text_feat, idx=idx)

            if self.mlm:
                attr_loss_mlm = self.get_mlm_loss_attr(attr_text_ids_masked, attr_text_atts, image_embeds, image_atts,
                                                       attr_masked_pos, attr_masked_ids, label)
                loss_mlm = self.get_mlm_loss(text_ids_masked, text_atts, image_embeds, image_atts, masked_pos,
                                             masked_ids)
                loss_attr = (attr_loss_itc + attr_loss_itm + attr_loss_mlm) / 3
                return loss_itc, loss_itm, loss_mlm, loss_attr
            else:
                loss_attr = (attr_loss_itc + attr_loss_itm) / 2
                return loss_itc, loss_itm, loss_attr

        image_embeds, image_atts = self.get_vision_embeds(image)
        text_embeds = self.get_text_embeds(text_ids, text_atts)
        image_feat, text_feat = self.get_features(image_embeds, text_embeds)
        # for weak supervision
        loss_itc = self.get_contrastive_loss(image_feat, text_feat, idx=idx)
        loss_itm = self.get_matching_loss(image_embeds, image_atts, image_feat,
                                          text_embeds, text_atts, text_feat, idx=idx)
        # eda
        if self.eda:
            add_loss = dict()
            text_embeds_eda = self.get_text_embeds(text_ids_eda, text_atts_eda)
            text_feat_eda = self.get_features(text_embeds=text_embeds_eda)
            loss_itc_eda = self.get_contrastive_loss(image_feat, text_feat_eda, idx=idx)
            loss_itm_eda = self.get_matching_loss(image_embeds, image_atts, image_feat,
                                                  text_embeds_eda, text_atts_eda, text_feat_eda, idx=idx)
            loss_itc = loss_itc + 0.8 * loss_itc_eda
            loss_itm = loss_itm + 0.8 * loss_itm_eda

            add_loss.update({'loss_itc': loss_itc})
            add_loss.update({'loss_itm': loss_itm})
        if self.apply_mlm:
            loss_mlm = self.get_mlm_loss(text_ids_masked, text_atts, image_embeds, image_atts, masked_pos, masked_ids)
            add_loss.update({'loss_mlm': loss_mlm})
        if self.lu:
            augmented_features = self.add_gaussian_noisy(image_feat)
            loss_lu = self.get_lu_loss(image_feat, text_feat, augmented_features, self.total_epoch, cur_epoch)
            add_loss.update({'loss_lu': loss_lu})
        if self.unc:
            loss_unc = self.get_unc_loss(image_feat, text_feat, idx, cur_epoch, self.total_epoch)
            add_loss.update({'loss_unc': loss_unc})
        if self.mim:
            mask_for_one_batch = self.build_masks_for_one_batch(image.shape[0])
            masked_img = self.build_masked_image(image, mask_for_one_batch)
            masked_img_embed, masked_img_atts = self.get_vision_embeds(masked_img)
            masked_img_feat = self.get_features(masked_img_embed, None)
            recon_image = self.mim_gen(masked_img_feat, text_feat)
            temp_image = self.get_unmasked_image(image, mask_for_one_batch).reshape([image.shape[0], 3*384*96])
            loss_mim = self.get_mim_loss(recon_image, temp_image)
            add_loss.update({'loss_mim': loss_mim})
        if self.sdm:
            loss_sdm = self.get_sdm_loss(image_feat, text_feat, logit_scale=self.temp, pid=idx)
            add_loss.update({'loss_sdm': loss_sdm})
        if self.id:
            image_logits = self.classfier(image_feat)
            text_logits = self.classfier(text_feat)
            loss_id = self.get_id_loss(image_logits, text_logits, idx)
            add_loss.update({'loss_id': loss_id})
        if self.match:
            loss_match = self.get_match_loss(image_feat, text_feat)
            add_loss.update({'loss_match': loss_match})

                
        return add_loss
