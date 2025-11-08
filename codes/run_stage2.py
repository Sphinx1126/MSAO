#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import logging
import os
import random
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import transformers


from HousE import HousEModel
from MSAO import MSAO

from torch.utils.data import DataLoader

import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from transfile import TransformerBlock
from collections import defaultdict
from operator import itemgetter
import pickle
import re
import time
import math
import networkx as nx

from processors import *
from DataloaderStage1 import TrainDataset, TestDataset
from DataloaderStage1 import BidirectionalOneShotIterator

import torch.distributed as dist
from torch.utils.data import BatchSampler
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from datetime import timedelta


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Models',
        usage='train.py [<args>] [-h | --help]'
    )

    parser.add_argument('--cuda', action='store_true', help='use GPU', default=True)
    parser.add_argument('--seed', default=10, type=int)
    
    parser.add_argument('--do_train', action='store_true', default=True)
    parser.add_argument('--do_valid', action='store_true', default=True)
    parser.add_argument('--do_test', action='store_true', default=True)
    parser.add_argument('--evaluate_train', action='store_true', default=False, help='Evaluate on training data')
    
    parser.add_argument('--countries', action='store_true', help='Use Countries S1/S2/S3 datasets')
    parser.add_argument('--regions', type=int, nargs='+', default=None, 
                        help='Region Id for Countries S1/S2/S3 datasets, DO NOT MANUALLY SET')
    
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--data_name', type=str, default='FB15k-237')
    parser.add_argument('--model', default='HousE_plus', type=str)
    parser.add_argument('-de', '--double_entity_embedding', action='store_true')
    parser.add_argument('-dr', '--double_relation_embedding', action='store_true')
    
    parser.add_argument('-n', '--negative_sample_size', default=256, type=int)
    parser.add_argument('-d', '--hidden_dim', default=512, type=int)
    parser.add_argument('-hd', '--house_dim', default=25, type=int)
    parser.add_argument('-hn', '--house_num', default=2, type=int)
    parser.add_argument('-dn', '--housd_num', default=6, type=int)
    parser.add_argument('-th', '--thred', default=0.6, type=float)
    parser.add_argument('-g', '--gamma', default=5.0, type=float)
    parser.add_argument('-adv', '--negative_adversarial_sampling', action='store_true', default=False)
    parser.add_argument('-a', '--adversarial_temperature', default=2.0, type=float)
    parser.add_argument('-t', '--loss_adversarial_temperature', default=2.0, type=float)
    parser.add_argument('-b', '--batch_size', default=512, type=int)
    parser.add_argument('-r', '--regularization', default=0.1, type=float)
    parser.add_argument('-e_reg', '--ent_reg', default=0.0, type=float)
    parser.add_argument('-r_reg', '--rel_reg', default=0.0, type=float)
    parser.add_argument('--test_batch_size', default=4, type=int, help='valid/test batch size')
    parser.add_argument('--uni_weight', action='store_true', 
                        help='Otherwise use subsampling weighting like in word2vec')
    
    parser.add_argument('-lr', '--learning_rate', default=0.0002, type=float)
    parser.add_argument('--lr_kge', default=2e-6, type=float)
    parser.add_argument('-cpu', '--cpu_num', default=10, type=int)
    parser.add_argument('-init', '--init_checkpoint', default=None, type=str)
    parser.add_argument('-save', '--save_path', default='../models/HousE_FB15k_0', type=str)
    parser.add_argument('--max_steps', default=75000, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--warm_up_steps', default=35000, type=int)
    
    parser.add_argument('--save_checkpoint_steps', default=100000, type=int)
    parser.add_argument('--valid_steps', default=5000, type=int)
    parser.add_argument('--log_steps', default=1000, type=int, help='train log every xx steps')
    parser.add_argument('--test_log_steps', default=1000, type=int, help='valid/test log every xx steps')
    
    parser.add_argument('--nentity', type=int, default=0, help='DO NOT MANUALLY SET')
    parser.add_argument('--nrelation', type=int, default=0, help='DO NOT MANUALLY SET')


    parser.add_argument('--use_RelaGraph', default=True, help='whether to add inverse edges')
    parser.add_argument('--inverse', action='store_true', help='whether to add inverse edges')
    parser.add_argument('--val_inverse', action='store_true', help='whether to add inverse edges to the validation set')
    parser.add_argument('--drop', type=float, default=0.1, help='Dropout in layers')
    parser.add_argument('--init_std', type=float, default=1e-5, help='Dropout in layers')
    parser.add_argument('--anchor_size', default=1.0, type=float, help='size of the anchor set, i.e. |A|')
    parser.add_argument('--drop_size', default=0.05, type=float, help='size of the anchor set, i.e. |A|')
    parser.add_argument('-ancs', '--sample_anchors', default=8, type=int)
    parser.add_argument('--node_dim', default=0, type=int)
    parser.add_argument('-merge', '--merge_strategy', default='mean_pooling', type=str,
                        help='how to merge information from anchors, chosen between [ mean_pooling, linear_proj ]')
    parser.add_argument('-layers', '--attn_layers_num', default=1, type=int)
    parser.add_argument('--mlp_ratio', default=0.2, type=float)
    parser.add_argument('--head_dim', default=8, type=int)
    parser.add_argument('-type', '--add_type_embedding', default=True)
    parser.add_argument('-share', '--anchor_share_embedding', default=True)
    
    return parser.parse_args(args)

def override_config(args):
    '''
    Override model and data configuration
    '''
    
    with open(os.path.join(args.init_checkpoint, 'config.json'), 'r') as fjson:
        argparse_dict = json.load(fjson)
    
    args.countries = argparse_dict['countries']
    if args.data_path is None:
        args.data_path = argparse_dict['data_path']
    args.model = argparse_dict['model']
    args.double_entity_embedding = argparse_dict['double_entity_embedding']
    args.double_relation_embedding = argparse_dict['double_relation_embedding']
    args.hidden_dim = argparse_dict['hidden_dim']
    args.test_batch_size = argparse_dict['test_batch_size']
    
def save_model(model, optimizer, save_variable_list, args, best_valid=False):
    '''
    Save the parameters of the model and the optimizer,
    as well as some other variables such as step and learning_rate
    '''
    
    argparse_dict = vars(args)

    if best_valid:
        save_path = args.save_path + "/best_model"
    else:
        save_path = args.save_path
    
    with open(os.path.join(save_path, 'config.json'), 'w') as fjson:
        json.dump(argparse_dict, fjson)

    torch.save({
        **save_variable_list,
        'model_state_dict': model.state_dict(),
        #'optimizer_state_dict': optimizer.state_dict()
        },
        os.path.join(save_path, 'checkpoint')
    )

def read_triple(file_path, entity2id, relation2id):
    '''
    Read triples and map them into ids.
    '''
    triples = []
    with open(file_path) as fin:
        for line in fin:
            h, r, t = line.strip().split('\t')
            triples.append((entity2id[h], relation2id[r], entity2id[t]))
    return triples

def set_logger(args):
    '''
    Write logs to checkpoint and console
    '''
    if not os.path.exists(args.save_path + '/Stage2'):
        os.makedirs(args.save_path + '/Stage2')
    if not os.path.exists(args.save_path + "/Stage2/best_model"):
        os.makedirs(args.save_path + "/Stage2/best_model")

    if args.do_train:
        log_file = os.path.join(args.save_path or args.init_checkpoint, 'Stage2/train.log')
    else:
        log_file = os.path.join(args.save_path or args.init_checkpoint, 'Stage2/test.log')

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

def log_metrics(mode, step, metrics):
    '''
    Print the evaluation logs
    '''
    for metric in metrics:
        logging.info('%s %s at step %d: %f' % (mode, metric, step, metrics[metric]))

def train_step(model, optimizer, scheduler, train_iterator, args, rank, stage = '1'):
    '''
    A single train step. Apply back-propation and return the loss
    '''

    model.train()

    optimizer.zero_grad()
    #optimizer.module.zero_grad()

    positive_sample, negative_sample, subsampling_weight, mode = next(train_iterator)
    if args.cuda:
        positive_sample = positive_sample.to(rank)
        negative_sample = negative_sample.to(rank)
        subsampling_weight = subsampling_weight.to(rank)
    
    dist.barrier()
    negative_score = model((positive_sample, negative_sample), mode=mode, rank=rank, stage = stage)
    if mode == 'head-batch':
        pos_part = positive_sample[:, 0].unsqueeze(dim=1)
    else:
        pos_part = positive_sample[:, 2].unsqueeze(dim=1)
    positive_score = model((positive_sample, pos_part), mode=mode, rank=rank, stage = stage)


    if args.negative_adversarial_sampling:
        #In self-adversarial sampling, we do not apply back-propagation on the sampling weight
        negative_score = (F.softmax(negative_score * args.adversarial_temperature, dim = 1).detach() 
                            * F.logsigmoid(-negative_score)).sum(dim = 1)
    else:
        negative_score = F.logsigmoid(-negative_score).mean(dim = 1)


    positive_score = F.logsigmoid(positive_score).squeeze(dim = 1)

    if args.uni_weight:
        positive_sample_loss = - positive_score.mean()
        negative_sample_loss = - negative_score.mean()
    else:
        positive_sample_loss = - (subsampling_weight * positive_score).sum()/subsampling_weight.sum()
        negative_sample_loss = - (subsampling_weight * negative_score).sum()/subsampling_weight.sum()

    weight = torch.exp(negative_sample_loss * args.loss_adversarial_temperature - positive_sample_loss * args.loss_adversarial_temperature).detach() 
    loss = (positive_sample_loss + weight * negative_sample_loss) / (1 + weight)

    if args.regularization != 0.0:
        regularization = args.regularization * (
            model.module.kge_model.entity_embedding.norm(dim=2, p=2).norm(dim=1, p=2).mean()
        )
        loss = loss + regularization
        regularization_log = {'regularization': regularization.item()}
    else:
        regularization_log = {}  

    dist.barrier()
    loss.backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
    
    log = {
        **regularization_log,
        'positive_sample_loss': positive_sample_loss.item(),
        'negative_sample_loss': negative_sample_loss.item(),
        'loss': loss.item()
    }

    return log

def test_step(model, test_triples, all_true_triples, args, rank, stage = '1'):
    '''
    Evaluate the model on test or valid datasets
    '''
    
    model.eval()
    
    if args.countries:
        #Countries S* datasets are evaluated on AUC-PR
        #Process test data for AUC-PR evaluation
        sample = list()
        y_true  = list()
        for head, relation, tail in test_triples:
            for candidate_region in args.regions:
                y_true.append(1 if candidate_region == tail else 0)
                sample.append((head, relation, candidate_region))

        sample = torch.LongTensor(sample)
        if args.cuda:
            sample = sample.to(rank)

        with torch.no_grad():
            y_score = model(sample, rank=rank, stage = stage).squeeze(1).cpu().numpy()

        y_true = np.array(y_true)

        #average_precision_score is the same as auc_pr
        auc_pr = average_precision_score(y_true, y_score)

        metrics = {'auc_pr': auc_pr}
        
    else:
        #Otherwise use standard (filtered) MRR, MR, HITS@1, HITS@3, and HITS@10 metrics
        #Prepare dataloader for evaluation
        test_dataset_head = TestDataset(test_triples, all_true_triples, args.nentity, args.nrelation, 'head-batch')
        test_dataloader_head = DataLoader(
            test_dataset_head, 
            #sampler = DistributedSampler(test_dataset_head, shuffle= False, seed = args.seed), 
            batch_size=args.test_batch_size,
            shuffle=False,
            num_workers=0, 
            collate_fn=TestDataset.collate_fn
        )

        test_dataset_tail = TestDataset(test_triples, all_true_triples, args.nentity, args.nrelation, 'tail-batch')
        test_dataloader_tail = DataLoader(
            test_dataset_tail,
            #sampler = DistributedSampler(test_dataset_tail, shuffle= False, seed = args.seed), 
            batch_size=args.test_batch_size,
            shuffle=False,
            num_workers=0, 
            collate_fn=TestDataset.collate_fn
        )
        
        test_dataset_list = [test_dataloader_head, test_dataloader_tail]
        
        logs = []

        step = 0
        total_steps = len(test_dataloader_head) + len(test_dataloader_tail)

        with torch.no_grad():
            for test_dataset in test_dataset_list:
                for data in test_dataset:
                    positive_sample, negative_sample, filter_bias, mode = data
                    if args.cuda:
                        positive_sample = positive_sample.to(rank)
                        negative_sample = negative_sample.to(rank)
                        filter_bias = filter_bias.to(rank)

                    batch_size = positive_sample.size(0)

                    score = model((positive_sample, negative_sample), mode=mode, rank=rank, stage = stage)
                    score += filter_bias

                    #Explicitly sort all the entities to ensure that there is no test exposure bias
                    argsort = torch.argsort(score, dim = 1, descending=True)

                    if mode == 'head-batch':
                        positive_arg = positive_sample[:, 0]
                    elif mode == 'tail-batch':
                        positive_arg = positive_sample[:, 2]
                    else:
                        raise ValueError('mode %s not supported' % mode)

                    for i in range(batch_size):
                        #Notice that argsort is not ranking
                        ranking = (argsort[i, :] == positive_arg[i]).nonzero()
                        assert ranking.size(0) == 1

                        #ranking + 1 is the true ranking used in evaluation metrics
                        ranking = 1 + ranking.item()
                        logs.append({
                            'MRR': 1.0/ranking,
                            'MR': float(ranking),
                            'HITS@1': 1.0 if ranking <= 1 else 0.0,
                            'HITS@3': 1.0 if ranking <= 3 else 0.0,
                            'HITS@10': 1.0 if ranking <= 10 else 0.0,
                        })

                    if step % args.test_log_steps == 0 and rank==0:
                        logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))

                    step += 1
        metrics = {}
        for metric in logs[0].keys():
            metrics[metric] = sum([log[metric] for log in logs])/len(logs)

    return metrics
        
def objective(rank, world_size, args):
    dist.init_process_group("nccl", timeout=timedelta(seconds=7200000), rank=rank, world_size=world_size)
    
    if rank==0:
        set_logger(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    
    if args.house_dim % 2 == 0:
        args.house_num = args.house_dim
    else:
        args.house_num = args.house_dim-1

    if (not args.do_train) and (not args.do_valid) and (not args.do_test):
        raise ValueError('one of train/val/test mode must be choosed.')
    
    if args.init_checkpoint:
        override_config(args)
    elif args.data_path is None:
        raise ValueError('one of init_checkpoint/data_path must be choosed.')

    if args.do_train and args.save_path is None:
        raise ValueError('Where do you want to save your trained model?')
    
    
    # Write logs to checkpoint and console
    # set_logger(args)
    with open(os.path.join(args.data_path, 'entities.dict')) as fin:
        entity2id = dict()
        for line in fin:
            eid, entity = line.strip().split('\t')
            entity2id[entity] = int(eid)

    with open(os.path.join(args.data_path, 'relations.dict')) as fin:
        relation2id = dict()
        for line in fin:
            rid, relation = line.strip().split('\t')
            relation2id[relation] = int(rid)
    
    # Read regions for Countries S* datasets
    if args.countries:
        regions = list()
        with open(os.path.join(args.data_path, 'regions.list')) as fin:
            for line in fin:
                region = line.strip()
                regions.append(entity2id[region])
        args.regions = regions

    nentity = len(entity2id)
    nrelation = len(relation2id)
    
    args.nentity = nentity
    args.nrelation = nrelation
    #max_steps = max(100000, args.max_steps)
    
    if rank==0:
        logging.info('Parameters: %s' % args)
        logging.info('Model: %s' % args.model)
        logging.info('Data Path: %s' % args.data_path)
        logging.info('#entity: %d' % nentity)
        logging.info('#relation: %d' % nrelation)
    
    train_triples = read_triple(os.path.join(args.data_path, 'train.txt'), entity2id, relation2id)
    if rank==0:
        logging.info('#train: %d' % len(train_triples))
    valid_triples = read_triple(os.path.join(args.data_path, 'valid.txt'), entity2id, relation2id)
    if rank==0:
        logging.info('#valid: %d' % len(valid_triples))
    test_triples = read_triple(os.path.join(args.data_path, 'test.txt'), entity2id, relation2id)
    if rank==0:
        logging.info('#test: %d' % len(test_triples))
    
    #All true triples
    all_true_triples = train_triples + valid_triples + test_triples

    kge_model = HousEModel(
        model_name=args.model,
        nentity=nentity,
        nrelation=nrelation,
        hidden_dim=args.hidden_dim,
        gamma=args.gamma,
        house_dim=args.house_dim,
        house_num=args.house_num,
        housd_num=args.housd_num,
        thred=args.thred,
        double_entity_embedding=args.double_entity_embedding,
        double_relation_embedding=args.double_relation_embedding
    )
    if rank==0:
        logging.info('Loading model of Stage 1...')
    checkpoint = torch.load(os.path.join(args.save_path, 'best_model/checkpoint'))
    kge_model.load_state_dict(checkpoint['model_state_dict'])
    model = MSAO(kge_model, train_triples, all_true_triples, args)
    
    args.save_path = args.save_path + '/Stage2'


    if rank==0:
        logging.info('Model Parameter Configuration:')
        for name, param in model.named_parameters():
            logging.info('Parameter %s: %s, require_grad = %s' % (name, str(param.size()), str(param.requires_grad)))

    if args.cuda:
        model = model.to(rank)
        if rank==0:
            logging.info('Num of GPU: %d' % torch.cuda.device_count())
    
    if args.do_train:
        # Set training dataloader iterator
        train_dataset_head = TrainDataset(train_triples, nentity, nrelation, args.negative_sample_size, 'head-batch')
        train_sampler_head = DistributedSampler(train_dataset_head, seed = args.seed)
        train_dataloader_head = DataLoader(
            train_dataset_head,
            sampler = train_sampler_head,
            batch_size = args.batch_size // torch.cuda.device_count(),
            num_workers=0,
            collate_fn=TrainDataset.collate_fn
        )
        
        train_dataset_tail = TrainDataset(train_triples, nentity, nrelation, args.negative_sample_size, 'tail-batch')
        train_sampler_tail = DistributedSampler(train_dataset_tail, seed = args.seed)
        train_dataloader_tail = DataLoader(
            train_dataset_tail,
            sampler = train_sampler_tail,
            batch_size = args.batch_size // torch.cuda.device_count(),
            num_workers=0,
            collate_fn=TrainDataset.collate_fn
        )
        
        train_iterator = BidirectionalOneShotIterator(train_dataloader_head, train_dataloader_tail)
        
        # Set training configuration
        optimizer = transformers.AdamW(
            [{
                'params': [param for name,param in model.named_parameters() if 'kge_model' in name], 
                'lr': args.lr_kge,
                'weight_decay': args.weight_decay
            },{
                'params': [param for name,param in model.named_parameters() if 'kge_model' not in name], 
                'lr': args.learning_rate, 
                'weight_decay': args.weight_decay
            }]
        )
        scheduler = transformers.get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warm_up_steps, num_training_steps=args.max_steps + 1
        )

    if rank==0:
        logging.info('Start Training...')
        logging.info('batch_size = %d' % args.batch_size)
        logging.info('negative_adversarial_sampling = %d' % args.negative_adversarial_sampling)
        logging.info('hidden_dim = %d' % args.hidden_dim)
        logging.info('gamma = %f' % args.gamma)
        logging.info('negative_adversarial_sampling = %s' % str(args.negative_adversarial_sampling))
        if args.negative_adversarial_sampling:
            logging.info('adversarial_temperature = %f' % args.adversarial_temperature)
    
    # Set valid dataloader as it would be evaluated during training
    
    if args.do_train:
        ddp_model = DDP(model, device_ids=[rank])#, find_unused_parameters=True

        training_logs = []
        best_mrr = 0.0
        stage = '2'
        
        #Training Loop
        for step in tqdm(range(args.max_steps + 1)):
            
            log = train_step(ddp_model, optimizer, scheduler, train_iterator, args, rank, stage = stage)
            
            training_logs.append(log)

            if step % args.save_checkpoint_steps == 0:
                save_variable_list = {
                    'step': step, 
                    #'current_learning_rate': current_learning_rate,
                    #'warm_up_steps': warm_up_steps
                }
                if rank==0:
                    save_model(model, optimizer, save_variable_list, args)
                
            if step % args.log_steps == 0:
                metrics = {}
                for metric in training_logs[0].keys():
                    metrics[metric] = sum([log[metric] for log in training_logs])/len(training_logs)
                if rank==0:
                    log_metrics('Training average', step, metrics)
                training_logs = []
            
            if rank==0 and args.do_valid and (step % args.valid_steps == 0 or step == args.max_steps) and step != 0:
                logging.info('Evaluating on Valid Dataset...')
                metrics = test_step(ddp_model, valid_triples, all_true_triples, args, rank, stage = stage)
                log_metrics('Valid', step, metrics)
                if metrics['MRR'] > best_mrr:
                    save_variable_list = {
                        'step': step, 
                        #'current_learning_rate': current_learning_rate,
                        #'warm_up_steps': warm_up_steps
                    }
                    if rank==0:
                        save_model(model, optimizer, save_variable_list, args, True)
                    best_mrr = metrics['MRR']
                
                    if args.do_test:
                        logging.info('Evaluating on Test Dataset...')
                        metrics = test_step(ddp_model, test_triples, all_true_triples, args, rank, stage = stage)
                        log_metrics('Test', step, metrics)
            
        
        save_variable_list = {
            'step': step, 
            #'current_learning_rate': current_learning_rate,
            #'warm_up_steps': warm_up_steps
        }
        if rank==0:
            save_model(model, optimizer, save_variable_list, args)

    if rank==0 and args.evaluate_train:
        logging.info('Evaluating on Training Dataset...')
        metrics = test_step(ddp_model, train_triples, all_true_triples, args, rank, stage = stage)
        log_metrics('Test', step, metrics)

    if rank==0 and args.do_valid:
        logging.info('Evaluating on Valid Dataset with Best Model...')
        checkpoint = torch.load(os.path.join(args.save_path + "/best_model", 'checkpoint'))
        best_step = checkpoint['step']
        model.load_state_dict(checkpoint['model_state_dict'])
        metrics = test_step(ddp_model, valid_triples, all_true_triples, args, rank, stage = stage)
        log_metrics('Valid', best_step, metrics)
    
    if rank==0 and args.do_test:
        logging.info('Evaluating on Test Dataset with Best Model...')
        checkpoint = torch.load(os.path.join(args.save_path + "/best_model", 'checkpoint'))
        best_step = checkpoint['step']
        model.load_state_dict(checkpoint['model_state_dict'])
        metrics = test_step(ddp_model, test_triples, all_true_triples, args, rank, stage = stage)
        log_metrics('Test', best_step, metrics)
        final_result = metrics['MRR']
    
    while True:
        time.sleep(5)

if __name__ == '__main__':
    args = parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "39601"
    os.environ['NCCL_BLOCKING_WAIT'] = '0'
    world_size = torch.cuda.device_count()

    if args.save_path and not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    if not os.path.exists(args.save_path + "/best_model"):
        os.makedirs(args.save_path + "/best_model")
    args.data_path=os.path.join(args.data_dir, args.data_name)

    mp.spawn(objective, args=(world_size,args,), nprocs=world_size, join=True)
