import sys
import os
import json
from functools import partial
from tqdm import tqdm
import time
import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import torchvision
import torchnet as tnt

from utils import filter_opt
import log as log_utils

# classification models
from classifiers.NCM import NearestClassMean
from classifiers.NCM_open import OpenNCM
from classifiers.NCM_openmax import NCMOpenMax
from classifiers.NCM_open import OpenNCM

from metrics import compute_metrics

def test_model(data_loader, classifier, unknow_id, force_unk_testdata=False):
    y_pred_tot = []
    y_true = []
    y_score = []
    y_pred_close_tot = []
    y_pred_ood_tot = []

    total_infer_time = 0.0
    total_samples = 0
    
    score_corr = 0
    score_wrong = 0

    for sample in tqdm(data_loader):
        x = sample['data']
        labels = sample['label'] # labels

        # replace labels with unknown
        if force_unk_testdata:
            labels = ['_unknown_' for item in labels]
        
        batch_size = len(labels)
        total_samples += batch_size        

        # perform classification
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.perf_counter()

        p_y, target_ids = classifier.evaluate_batch(x, labels, return_probas=False)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time
        total_infer_time += elapsed

        conf_val, y_pred = p_y.max(1)

        if '_unknown_' in classifier.word_to_index.keys():
            y_pred_ood = p_y[:,unknow_id]
            y_pred_close = np.delete(p_y, unknow_id, 1)
        else:
            y_pred_close = p_y
            y_pred_ood = np.array([])

        y_pred_tot +=  y_pred.tolist()
        y_pred_ood_tot += y_pred_ood.tolist()
        y_pred_close_tot += y_pred_close.tolist()

        target_ids = target_ids.squeeze().tolist()
        y_true += [target_ids] if isinstance(target_ids, int) else target_ids

        conf_val = conf_val.tolist()
        y_score += [conf_val] if isinstance(conf_val, int) else conf_val

    avg_infer_time_per_sample = total_infer_time / total_samples if total_samples > 0 else 0.0

    return (
        y_score,
        y_pred_tot,
        y_true,
        y_pred_close_tot,
        y_pred_ood_tot,
        avg_infer_time_per_sample,
        total_infer_time,
        total_samples,
    )


if __name__ == '__main__':
    from parser_kws import *
    args = parser.parse_args()
    opt = vars(parser.parse_args())

    os.environ['CUDA_VISIBLE_DEVICES'] = opt['cuda_idx']

    # load the encoder
    if os.path.isfile(opt['model.model_path']):    
        enc_model = torch.load(opt['model.model_path'], weights_only=False)
    else:
        raise ValueError("Model {} not valid".format(opt['model.model_path']))


    # load the classifier
    print('Using the classifier: ', opt['fsl.classifier'])
    if opt['fsl.classifier'] == 'ncm':
        classifier = NearestClassMean(backbone=enc_model, cuda=opt['data.cuda'])
    elif opt['fsl.classifier'] == 'openncm':
        classifier = OpenNCM(backbone=enc_model, cuda=opt['data.cuda'])
    elif opt['fsl.classifier'] == 'ncm_openmax':
        classifier = NCMOpenMax(backbone=enc_model, cuda=opt['data.cuda'])
    elif opt['fsl.classifier'] == 'open_nmc':
        classifier = OpenNCM(backbone=enc_model, cuda=opt['data.cuda'])
    else:
        raise ValueError("Classifier {} is not valid".format(opt['fsl.classifier']))

    #print(classifier)

    # import tasks: positive samples and optionative negative samples for open set
    # current limitations: tasks belongs to same dataset (separate eyword split)
    speech_args = filter_opt(opt, 'speech')
    dataset = opt['speech.dataset']
    data_dir = opt['speech.default_datadir'] 
    task = opt['speech.task'] 
    tasks = task.split(",")
    if len(tasks) == 2:
        pos_task, neg_task = tasks
    elif len(tasks) == 1:
        pos_task = tasks[0]
        neg_task = None

    if dataset == 'googlespeechcommand':
        from data.GSCSpeechData import GSCSpeechDataset
        ds = GSCSpeechDataset(data_dir, pos_task, opt['data.cuda'], speech_args)
        num_classes = ds.num_classes()
        opt['model.num_classes'] = num_classes
        print("The task {} of the {} Dataset has {} classes".format(
                pos_task, dataset, num_classes))
        
        ds_neg = None
        if neg_task is not None:
            ds_neg = GSCSpeechDataset(data_dir, neg_task, 
                    opt['data.cuda'], speech_args)
            print("The task {} is used for negative samples".format(
                    neg_task))       
    elif dataset == 'MSWC':
        from data.MSWCtest import MSWCDataset
        ds = MSWCDataset(data_dir, pos_task, opt['data.cuda'], speech_args)
        ds_neg = None
        if neg_task is not None:
            ds_neg = MSWCDataset(data_dir, neg_task, opt['data.cuda'], speech_args)
    else:
        raise ValueError("Dataset not recognized")

    # Few-Shot Parameters to configure the classifier for testing
    # the test is done over n_episodes
    # n_support support samples of n_way classes are avaible at test time 
    n_way = opt['fsl.test.n_way']
    n_support = opt['fsl.test.n_support']
    n_episodes = opt['fsl.test.n_episodes']
    fixed_silence_unknown = opt['fsl.test.fixed_silence_unknown']
    
    # setup dataloader of support samples
    # support samples are retrived from the training split of the dataset
    # if include_unknown is True, the _unknown_ class is one of the num_classes
    torch.manual_seed(17)
    train_episodic_loader = ds.get_episodic_dataloader('training', n_way, n_support, n_episodes)
    

    # Postprocess arguments
    #   list of log variables. may be turned into a configurable list usign opt['log.fields'] as 
    #   opt['log.fields'] = opt['log.fields'].split(',')
    opt['log.fields'] = [
        'aucROC',
        'accuracy_pos',
        'accuracy_neg',
        'acc_prec95',
        'frr_prec95',
        'avg_inference_time_pos_per_sample_sec'
    ]

    # import stats
    meters = { field: tnt.meter.AverageValueMeter() for field in opt['log.fields'] } 

        
    # test the model on multiple episodes 
    print("Evaluating model {} in a few-shot setting ({}-way | {}-shots) for {} episodes on the task {} of the Dataset {}".format(
            classifier.backbone.encoder.__class__.__name__, n_way,n_support, n_episodes, task, dataset))
    output = {'test':{}}
    for ep, support_sample in enumerate(train_episodic_loader):

        '''
            Classifier setup 
        '''
        # compute prototypes
        support_samples = support_sample['data']
        # extract label list          
        class_list = support_sample['label'][0]
        #print(class_list)
        # fit the classifier on the support samples
        classifier.fit_batch_offline(support_samples, class_list)
        #get the index of the unknown class of the classifier
        unk_idx = classifier.word_to_index.get('_unknown_', None)

        '''
            Few-shot test in open set
            NB: _unknown_ is the negative class as part of the class_list
        '''  
        # test on positive dataset     
        print('\n Test Episode {} with classes: {}'.format(ep, class_list))

        # load only samples from the target classes and not negative _unknown_
        query_loader = ds.get_iid_dataloader('testing', opt['fsl.test.batch_size'], 
            class_list = [x for x in class_list if 'unknown' not in x])

        (
            y_score_pos,
            y_pred_pos,
            y_true_pos,
            y_pred_close_pos,
            y_pred_ood_pos,
            avg_time_pos,
            total_time_pos,
            total_samples_pos,
        ) = test_model(query_loader, classifier, unk_idx)

        # test on the negative dataset (_unknown_) if present    
        if ds_neg is not None:
            neg_loader = ds_neg.get_iid_dataloader('testing', opt['fsl.test.batch_size'])
            (
                y_score_neg,
                y_pred_neg,
                y_true_neg,
                y_pred_close_neg,
                y_pred_ood_neg,
                avg_time_neg,
                total_time_neg,
                total_samples_neg,
            ) = test_model(neg_loader, classifier, unk_idx, force_unk_testdata=True)
        else:
            y_score_neg, y_pred_neg, y_true_neg, y_pred_close_neg, y_pred_ood_neg = None, None, None, None, None
            avg_time_neg, total_time_neg, total_samples_neg = 0.0, 0.0, 0

        # store and print metrics
        output_ep = compute_metrics(y_score_pos, y_pred_pos, y_true_pos, y_pred_close_pos, y_pred_ood_pos,
                        y_score_neg, y_pred_neg, y_true_neg, y_pred_close_neg, y_pred_ood_neg,
                        classifier.word_to_index, verbose=True)
        output_ep["avg_inference_time_pos_per_sample_sec"] = avg_time_pos
        output_ep["total_inference_time_pos_sec"] = total_time_pos
        output_ep["num_pos_samples"] = total_samples_pos

        print(f"Episode {ep} positive avg inference time per sample: {avg_time_pos:.6f} sec/sample")
        print(f"Episode {ep} positive total inference time: {total_time_pos:.6f} sec for {total_samples_pos} samples")    

        for field, meter in meters.items():
            if field in output_ep:
                meter.add(output_ep[field])

        output[str(ep)] = output_ep

    for field,meter in meters.items():
        mean, std = meter.value()
        output["test"][field] = {}
        output["test"][field]["mean"] = mean
        output["test"][field]["std"] = std
        print("Final Test: Avg {} is {:.5f} with std dev {:.5f}".format(field, mean, std))

    # write log
    if speech_args['include_unknown']:
        n_way = n_way - 1
    if classifier.backbone.emb_norm:
        fsl_z_norm = "NORM"
    else: 
        fsl_z_norm = "NOTN"
    
    
    output_log_file = 'eval{}_fsl_{}_{}shot'.format(
        opt['fsl.test.note'], opt['fsl.classifier'], n_support
    )
    results_dir = os.path.join(os.path.dirname(opt['model.model_path']), "multi_teacher")
    os.makedirs(results_dir, exist_ok=True)

    output_file = os.path.join(results_dir, output_log_file)
    print('Writing log to:', output_file)

    with open(output_file, 'w') as fp:
        print(f'{n_episodes} EPISODES', file=fp)
        json.dump(output, fp, indent=2)