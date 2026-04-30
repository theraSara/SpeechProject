import os
import json
from tqdm import tqdm
import time
import numpy as np
from shutil import copyfile

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchnet as tnt

from utils import filter_opt
import log as log_utils


if __name__ == '__main__':

    # read and post-process options
    from parser_kws import *
    args = parser.parse_args()
    opt = vars(parser.parse_args())

    os.environ['CUDA_VISIBLE_DEVICES'] = opt['cuda_idx']

    opt['log.fields'] = ['loss']
    speech_args = filter_opt(opt, 'speech')
    teacher_opt = filter_opt(opt, 'teacher')

    n_way = opt['train.n_way']
    n_support = opt['train.n_support']
    n_query = opt['train.n_query']
    n_episodes = opt['train.n_episodes']


    from models.ssl_teacher import SSLTeacher
    model = SSLTeacher(
        model_name=teacher_opt['model_name'],
        out_dim=teacher_opt['out_dim'],
        lora_r=teacher_opt['lora_r'],
        lora_alpha=teacher_opt['lora_alpha'],
        margin=opt['train.margin'],
    )
    model.backbone.print_trainable_parameters()

    if opt['data.cuda']:
        model.cuda()



    dataset = opt['speech.dataset']
    data_dir = opt['speech.default_datadir']
    train_task = opt['speech.task']

    from data.MSWCData import MSWCDataset
    ds_tr = MSWCDataset(data_dir, train_task, False, speech_args)

    num_classes_tr = ds_tr.num_classes()
    print("The training task {} of the {} Dataset has {} classes".format(dataset, train_task, num_classes_tr))
    n_way_tr = min(max(n_way, 0), num_classes_tr)


    meters = { 'train': { field: tnt.meter.AverageValueMeter() for field in opt['log.fields'] } }

    optim_method = getattr(optim, opt['train.optim_method'])
    optim_config = { 'lr': opt['train.learning_rate'],
                     'weight_decay': opt['train.weight_decay'] }
    optimizer = optim_method(model.parameters(), **optim_config)

    scheduler = lr_scheduler.StepLR(optimizer, opt['train.decay_every'], gamma=0.5)

    opt['log.exp_dir'] = os.path.join('./results', opt['log.exp_dir'])
    trace_file = os.path.join(opt['log.exp_dir'], 'trace.txt')
    checkpoint_file = os.path.join(opt['log.exp_dir'], 'checkpoint.pt')


    if os.path.isfile(checkpoint_file):
        print('Found Checkpoint!')
        checkpoint = torch.load(checkpoint_file, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['start_epoch']
        best_loss = checkpoint['best_loss']
        wait = checkpoint['wait']
        start_episode = checkpoint['start_episode']
        meters = checkpoint['meters']
    else:
        if not os.path.isdir(opt['log.exp_dir']):
            os.makedirs(opt['log.exp_dir'])
        #trace file
        if os.path.isfile(trace_file):
            os.remove(trace_file)

        # save opts
        with open(os.path.join(opt['log.exp_dir'], 'opt.json'), 'w') as f:
            json.dump(opt, f)
            f.write('\n')

        start_epoch = 0
        best_loss = np.inf
        wait = 0
        start_episode = 0
        meters = { 'train': { field: tnt.meter.AverageValueMeter() for field in opt['log.fields'] } }



    max_epoch = opt['train.epochs']
    cuda = opt['data.cuda']
    stop = False
    epoch = start_epoch

    model.train()

    while epoch < max_epoch and not stop:
        # get episode loaders
        episodic_loader = ds_tr.get_episodic_dataloader('training', n_way_tr,
            n_support+n_query, n_episodes-start_episode)

        if start_episode == 0:
            for split, split_meters in meters.items():
                for field, meter in split_meters.items():
                    meter.reset()

        ep_idx = start_episode
        for samples in tqdm(episodic_loader, desc="Epoch {:d} train".format(epoch + 1)):
            samples_ep = samples['data']
            if cuda:
                samples_ep = samples_ep.cuda()
            optimizer.zero_grad()
            loss, output = model.loss(samples_ep)
            loss.backward()
            optimizer.step()

            for field, meter in meters['train'].items():
                meter.add(output[field])

            ep_idx+=1

            # save checkpoint every 1/5 of total episodes
            stored_ckpt = False
            if ep_idx % (n_episodes//5) == 0 or ep_idx == n_episodes:
                checkpoint_file_tmp = os.path.join(opt['log.exp_dir'], 'checkpoint_tmp.pt')
                while stored_ckpt is False:
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'start_epoch': epoch,
                        'start_episode': ep_idx,
                        'best_loss': best_loss,
                        'wait': wait,
                        'meters': meters
                        }, checkpoint_file_tmp)
                    try:
                        torch.load(checkpoint_file_tmp, weights_only=False)
                    except EOFError:
                        print('Error Storing Ckpt at episode {} of epoch {}'.format(
                                ep_idx, epoch))
                    else:
                        copyfile(checkpoint_file_tmp, checkpoint_file)
                        stored_ckpt = True

        # end epoch
        if start_episode < n_episodes: scheduler.step()
        start_episode = 0

        # calculate loss on test set
        meters['test'] = { field: tnt.meter.AverageValueMeter() for field in opt['log.fields'] }
        test_loader = ds_tr.get_episodic_dataloader('testing', n_way_tr, 10, n_episodes)
        for samples in tqdm(test_loader, desc="Epoch {:d} test".format(epoch + 1)):
            samples_ep = samples['data']
            if cuda:
                samples_ep = samples_ep.cuda()
            with torch.no_grad():
                loss, output = model.loss(samples_ep)
            for field, meter in meters['test'].items():
                meter.add(output[field])

        # log at the end of the epoch
        meter_vals = log_utils.extract_meter_values(meters)
        print("Epoch {:d}: {:s}".format(epoch+1, log_utils.render_meter_values(meter_vals)))
        meter_vals['epoch'] = epoch+1
        with open(trace_file, 'a') as f:
            json.dump(meter_vals, f)
            f.write('\n')

        model.cpu()
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_name': teacher_opt['model_name'],
            'out_dim': teacher_opt['out_dim'],
            'lora_r': teacher_opt['lora_r'],
            'lora_alpha': teacher_opt['lora_alpha'],
            'margin': opt['train.margin'],
        }, os.path.join(opt['log.exp_dir'], 'best_model.pt'))
        if cuda:
            model.cuda()

        del meters['test']
        epoch += 1
