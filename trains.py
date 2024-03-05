import utils
from train_tools import mlm
import numpy as np


def train(model, data_loader, optimizer, tokenizer, epoch, device, scheduler, config, mask_generator=None):
    model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_itc', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_itm', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    if config['apply_mlm']:
        metric_logger.add_meter('loss_mlm', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    if config['lu']:
        metric_logger.add_meter('loss_lu', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    if config['unc']:
        metric_logger.add_meter('loss_unc', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    if config['mim']:
        metric_logger.add_meter('loss_mim', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    if config['sdm']:
        metric_logger.add_meter('loss_sdm', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    if config['id']:
        metric_logger.add_meter('loss_id', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    if config['match']:
        metric_logger.add_meter('loss_match', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50

    if config['eda']:
        for i, (image, text, text_eda, idx) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
            image = image.to(device, non_blocking=True)
            idx = idx.to(device, non_blocking=True)
            # text_input = tokenizer(text, padding='longest', max_length=config['max_tokens'],
            #                        return_tensors="pt").to(device)
            text_input = tokenizer(text, padding='max_length', truncation=True, max_length=config['max_tokens'],
                                   return_tensors="pt").to(device)
            text_input_eda = tokenizer(text_eda, padding='max_length', truncation=True, max_length=config['max_tokens'],
                                       return_tensors="pt").to(device)
            # if config['mlm']:
                # if config['lu']:
            text_ids_masked, masked_pos, masked_ids = mlm(text, text_input, tokenizer, device, mask_generator, config)
            if config['weak_supervision']:
                add_loss= model(image, text_input.input_ids, text_input.attention_mask,
                                                    text_ids_masked=text_ids_masked,
                                                    masked_pos=masked_pos, masked_ids=masked_ids, idx=None,
                                                    text_ids_eda=text_input_eda.input_ids,
                                                    text_atts_eda=text_input_eda.attention_mask,cur_epoch=epoch)
            else:        
                add_loss= model(image, text_input.input_ids, text_input.attention_mask,
                                                    text_ids_masked=text_ids_masked,
                                                    masked_pos=masked_pos, masked_ids=masked_ids, idx=idx,
                                                    text_ids_eda=text_input_eda.input_ids,
                                                    text_atts_eda=text_input_eda.attention_mask,cur_epoch=epoch)
            loss = sum([v for v in add_loss.values()])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            metric_logger.update(loss_itc=add_loss['loss_itc'].item())
            metric_logger.update(loss_itm=add_loss['loss_itm'].item())
            if config['apply_mlm']:
                metric_logger.update(loss_mlm=add_loss['loss_mlm'].item())
            if config['lu']:
                metric_logger.update(loss_lu=add_loss['loss_lu'].item())
            if config['unc']:
                metric_logger.update(loss_unc=add_loss['loss_unc'].item())
            if config['mim']:
                metric_logger.update(loss_mim=add_loss['loss_mim'].item())
            if config['id']:
                metric_logger.update(loss_id=add_loss['loss_id'].item())
            if config['sdm']:
                metric_logger.update(loss_sdm=add_loss['loss_sdm'].item())
            if config['match']:
                metric_logger.update(loss_match=add_loss['loss_match'].item())
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: "{:.5f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}