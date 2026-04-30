import os
import json
from pathlib import Path
import torch
import torchnet as tnt
import numpy as np
import classifiers
from tqdm import tqdm
from loguru import logger

from utils import filter_opt
from metrics import compute_metrics


def test_model(data_loader, classifier, unknow_id, force_unk_testdata=False):
    y_pred_tot = []
    y_true = []
    y_score = []
    y_pred_close_tot = []
    y_pred_ood_tot = []

    score_corr = 0
    score_wrong = 0

    for sample in tqdm(data_loader):
        x = sample['data']
        labels = sample['label'] # labela

        # replace labels with unknown
        if force_unk_testdata:
            labels = ['_unknown_' for item in labels]

        # perform classification
        p_y, target_ids = classifier.evaluate_batch(x, labels, return_probas=False)
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

    return y_score, y_pred_tot, y_true, y_pred_close_tot, y_pred_ood_tot


if __name__ == '__main__':
    from parser_kws import parser
    args = parser.parse_args()
    opt = vars(parser.parse_args())
    os.environ['CUDA_VISIBLE_DEVICES'] = opt['cuda_idx']

    # load the encoder
    model_path = Path(opt['model.model_path'])
    if model_path.is_file():
        enc_model = torch.load(model_path, weights_only=False)
    else:
        raise ValueError(f"Model {model_path} not valid")

    # load the classifier
    _name_mapping = {
        'ncm': 'NearestClassMean',
        'ncm_openmax': 'NCMOpenMax',
        'peeler': 'PeelerClass',
        'dproto': 'DProto',
    }
    classifier_name = opt['fsl.classifier']
    if classifier_name not in _name_mapping:
        raise ValueError(f'{classifier_name} not available')

    logger.info(f'Using the classifier: {classifier_name}')
    classifier = getattr(classifiers, _name_mapping[classifier_name])(
        backbone=enc_model, cuda=opt['data.cuda']
    )

    # import tasks: positive samples and optionative negative samples for open set
    # current limitations: tasks belongs to same dataset (separate keyword split)
    speech_args = filter_opt(opt, 'speech')
    task = speech_args['task']
    tasks = task.split(',')
    if len(tasks) >= 2:
        pos_task, neg_task, *_ = tasks
    else:
        pos_task, neg_task = tasks[0], None

    dataset = speech_args['dataset']
    data_dir = speech_args['datadir']
    logger.info(f'Using the dataset {dataset} with task {task}')
    if dataset == 'gsc':
        logger.debug('Have not optimize GSC')
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
    elif dataset == 'mswc':
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
    batch_size = opt['fsl.test.batch_size']

    logger.info(
        f"Evaluating model {classifier.backbone.encoder.__class__.__name__} "
        f"in a few-shot setting ({n_way}-way | {n_support}-shots) for {n_episodes} episodes "
        f"on the task {task} of the Dataset {dataset}"
    )

    # setup output logs
    log_fields = opt['log.fields'].split(',')
    meters = { field: tnt.meter.AverageValueMeter() for field in log_fields }
    OUTPUT = {}

    # setup dataloader of support samples
    # support samples are retrived from the training split of the dataset
    # if include_unknown is True, the _unknown_ class is one of the num_classes
    torch.manual_seed(17)
    train_episodic_loader = ds.get_episodic_dataloader('training', n_way, n_support, n_episodes)

    for ep, support_sample in enumerate(train_episodic_loader):
        support_samples = support_sample['data']
        class_list = support_sample['label'][0]
        classifier.fit_batch_offline(support_samples, class_list)
        unk_idx = classifier.word_to_index.get('_unknown_', None)

        # test on positive dataset
        logger.info(f'Test Episode {ep} with classes: {class_list}')

        # load only samples from the target classes and not negative _unknown_
        query_loader = ds.get_iid_dataloader('testing', batch_size,
            class_list = [x for x in class_list if 'unknown' not in x])

        y_score_pos, y_pred_pos, y_true_pos, y_pred_close_pos, y_pred_ood_pos = test_model(query_loader, classifier, unk_idx)

        # test on the negative dataset (_unknown_) if present
        if ds_neg is not None:
            neg_loader = ds_neg.get_iid_dataloader('testing', batch_size)
            y_score_neg, y_pred_neg, y_true_neg, y_pred_close_neg, y_pred_ood_neg = test_model(neg_loader, classifier, unk_idx, force_unk_testdata=True)
        else:
            y_score_neg, y_pred_neg, y_true_neg, y_pred_close_neg, y_pred_ood_neg = None, None, None, None, None

        # store and print metrics
        output_ep = compute_metrics(y_score_pos, y_pred_pos, y_true_pos, y_pred_close_pos, y_pred_ood_pos,
                        y_score_neg, y_pred_neg, y_true_neg, y_pred_close_neg, y_pred_ood_neg,
                        classifier.word_to_index, verbose=True)

        for field, meter in meters.items():
            meter.add(output_ep[field])
        OUTPUT[str(ep)] = output_ep


    OUTPUT['test'] = {}
    for field, meter in meters.items():
        OUTPUT["test"][field] = {}

        mean, std = meter.value()
        OUTPUT["test"][field]["mean"] = mean
        OUTPUT["test"][field]["std"] = std
        logger.info(f"Final Test: Avg {field} is {mean:.5f} with std dev {std:.5f}")

    # write log
    if speech_args['include_unknown']: n_way = n_way - 1
    fsl_z_norm = "NORM" if classifier.backbone.emb_norm else "NOTN"

    note = opt['log.note']
    log_file = (
        f"eval_fsl.{note}.{classifier_name}_{fsl_z_norm}."
        f"{n_way}way_{n_support}shot_{n_episodes}eps"
    )
    log_file = model_path.parent / log_file
    logger.info(f'Writing log to: {log_file}')

    with open(log_file, 'w') as fp:
        print(f'{n_episodes} EPISODES', file=fp)
        json.dump(OUTPUT, fp, indent=2)
