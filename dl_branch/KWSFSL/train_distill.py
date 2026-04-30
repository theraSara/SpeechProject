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
import models
from models.utils import get_model
import log as log_utils
from models.losses.distillation import p2p_loss, s2s_loss, kl_loss, compute_teacher_weights


if __name__ == '__main__':

    # read and post-process options
    from parser_kws import *
    args = parser.parse_args()
    opt = vars(parser.parse_args())

    os.environ['CUDA_VISIBLE_DEVICES'] = opt['cuda_idx']

    opt['model.x_dim'] = list(map(int, opt['model.x_dim'].split(',')))
    opt['log.fields'] = ['loss']
    speech_args = filter_opt(opt, 'speech')
    model_opt = filter_opt(opt, 'model')
    distill_opt = filter_opt(opt, 'distill')

    n_way = opt['train.n_way']
    n_support = opt['train.n_support']
    n_query = opt['train.n_query']
    n_episodes = opt['train.n_episodes']



    # prepare preprocessing
    if opt['model.preprocessing'] == 'mfcc':
        model_opt['mfcc'] = {
            'window_size_ms': speech_args['window_size'],
            'window_stride_ms': speech_args['window_stride'],
            'sample_rate': speech_args['sample_rate'],
            'n_mfcc': speech_args['n_mfcc'],
            'feature_bin_count': speech_args['num_features']
        }

    model_opt['loss'] = {'type': opt['train.loss'], 'margin':  opt['train.margin']}

    if os.path.isfile(opt['model.model_path']):
        print('Load Pretrained Student from', opt['model.model_path'])
        student = torch.load(opt['model.model_path'], weights_only=False)
        student.encoder.return_feat_maps = False
    else:
        print('Initializing new student')
        student = get_model(model_opt)

    # load teachers (frozen)
    from models.ssl_teacher import SSLTeacher
    teacher_paths = distill_opt['teacher_paths']
    teachers = []
    for path in teacher_paths:
        print('Load Teacher from', path)
        t = SSLTeacher.load(path)
        t.eval()
        for p in t.parameters():
            p.requires_grad_(False)
        teachers.append(t)

    # move to cuda
    cuda = opt['data.cuda']
    if cuda:
        student.cuda()
        for t in teachers:
            t.cuda()
        if 'mfcc' in model_opt.keys():
            student.preprocessing.mfcc.cuda()



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
    optimizer = optim_method(student.parameters(), **optim_config)

    scheduler = lr_scheduler.StepLR(optimizer, opt['train.decay_every'], gamma=0.5)

    opt['log.exp_dir'] = os.path.join('./results', opt['log.exp_dir'])
    trace_file = os.path.join(opt['log.exp_dir'], 'trace.txt')
    checkpoint_file = os.path.join(opt['log.exp_dir'], 'checkpoint.pt')



    if os.path.isfile(checkpoint_file):
        print('Found Checkpoint!')
        checkpoint = torch.load(checkpoint_file, weights_only=False)
        student.load_state_dict(checkpoint['model_state_dict'])
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



    alpha_tl  = distill_opt['alpha_tl']
    alpha_p2p = distill_opt['alpha_p2p']
    alpha_s2s = distill_opt['alpha_s2s']
    alpha_kl  = distill_opt['alpha_kl']
    kl_temperature = distill_opt['kl_temperature']

    max_epoch = opt['train.epochs']
    stop = False
    epoch = start_epoch

    student.train()

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
            x = samples['data']
            n_class_ep = x.size(0)
            n_sample_ep = x.size(1)
            x_flat = x.view(n_class_ep * n_sample_ep, *x.size()[2:])
            if cuda:
                x_flat = x_flat.cuda()

            optimizer.zero_grad()

            z_s = student.get_embeddings(x_flat)
            with torch.no_grad():
                z_ts = [t.get_embeddings(x_flat) for t in teachers]

            weights = compute_teacher_weights(z_ts, n_class_ep, n_sample_ep)

            loss = 0.0
            if alpha_tl  > 0: loss = loss + alpha_tl  * student.criterion.compute(z_s, n_sample_ep, n_class_ep)
            for w, z_t in zip(weights, z_ts):
                teacher_kd = 0.0
                if alpha_p2p > 0: teacher_kd = teacher_kd + alpha_p2p * p2p_loss(z_s, z_t)
                if alpha_s2s > 0: teacher_kd = teacher_kd + alpha_s2s * s2s_loss(z_s, z_t, n_sample_ep, n_class_ep)
                if alpha_kl  > 0: teacher_kd = teacher_kd + alpha_kl  * kl_loss(z_s, z_t, n_sample_ep, n_class_ep, kl_temperature)
                loss = loss + w * teacher_kd

            loss.backward()
            optimizer.step()

            meters['train']['loss'].add(loss.item())
            ep_idx+=1

            # save checkpoint every 1/5 of total episodes
            stored_ckpt = False
            if ep_idx % (n_episodes//5) == 0 or ep_idx == n_episodes:
                checkpoint_file_tmp = os.path.join(opt['log.exp_dir'], 'checkpoint_tmp.pt')
                while stored_ckpt is False:
                    torch.save({
                        'model_state_dict': student.state_dict(),
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
            x = samples['data']
            n_class_ep = x.size(0)
            n_sample_ep = x.size(1)
            x_flat = x.view(n_class_ep * n_sample_ep, *x.size()[2:])
            if cuda:
                x_flat = x_flat.cuda()
            with torch.no_grad():
                z_s = student.get_embeddings(x_flat)
                z_ts = [t.get_embeddings(x_flat) for t in teachers]
                weights = compute_teacher_weights(z_ts, n_class_ep, n_sample_ep)
                loss = 0.0
                if alpha_tl  > 0: loss = loss + alpha_tl  * student.criterion.compute(z_s, n_sample_ep, n_class_ep)
                for w, z_t in zip(weights, z_ts):
                    teacher_kd = 0.0
                    if alpha_p2p > 0: teacher_kd = teacher_kd + alpha_p2p * p2p_loss(z_s, z_t)
                    if alpha_s2s > 0: teacher_kd = teacher_kd + alpha_s2s * s2s_loss(z_s, z_t, n_sample_ep, n_class_ep)
                    if alpha_kl  > 0: teacher_kd = teacher_kd + alpha_kl  * kl_loss(z_s, z_t, n_sample_ep, n_class_ep, kl_temperature)
                    loss = loss + w * teacher_kd
            meters['test']['loss'].add(loss.item())

        # log at the end of the epoch
        meter_vals = log_utils.extract_meter_values(meters)
        print("Epoch {:d}: {:s}".format(epoch+1, log_utils.render_meter_values(meter_vals)))
        meter_vals['epoch'] = epoch+1
        with open(trace_file, 'a') as f:
            json.dump(meter_vals, f)
            f.write('\n')

        student.cpu()
        torch.save(student, os.path.join(opt['log.exp_dir'], 'best_model.pt'))
        if cuda:
            student.cuda()

        del meters['test']
        epoch += 1