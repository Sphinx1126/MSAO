# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 15:18:55 2025

@author: 28257
"""

#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from HousE import HousEModel

from sklearn.metrics import average_precision_score

from torch.utils.data import DataLoader
from DataloaderStage2 import TestDataset
from transfile import TransformerBlock

from collections import defaultdict
from operator import itemgetter
import pickle
import os
import re
import time
import math
import networkx as nx
from tqdm import tqdm

def no_error_listdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return os.listdir(path)

class MSAO(nn.Module):
    def __init__(self, kge_model, 
                 train_triples, all_true_triples, args,
                 double_entity_embedding=False, double_relation_embedding=False):
        super(MSAO, self).__init__()
        self.data_name = args.data_name
        self.args = args
        
        self.kge_model = kge_model
        '''
        if self.data_name == 'FB15k-237':
            self.kge_model.entity_embedding.requires_grad = False
            
            self.kge_model.relation_embedding.requires_grad = False
            self.kge_model.k_dir_head.requires_grad = False
            self.kge_model.k_dir_tail.requires_grad = False
            self.kge_model.k_scale_head.requires_grad = False
            self.kge_model.k_scale_tail.requires_grad = False
            self.kge_model.relation_weight.requires_grad = False
        '''

        self.train_triples = train_triples
        self.all_true_triples = all_true_triples

        self.dropout = nn.Dropout(0.05)
        self.dropout2 = nn.Dropout(0.1)

        anchor_dim = self.kge_model.entity_dim * self.kge_model.house_dim
        self.sample_anchors = args.sample_anchors

        type_ids = [] # token type embedding, center or neighbor or anchor

        type_ids.append(0)

        if self.sample_anchors > 0:
            type_ids.extend([2] * self.sample_anchors)
            print("Creating anchor infomation")
            self.hashes, self.anc_mask = self.tokenize_anchors(args.anchor_size, args.drop_size, self.sample_anchors)

        if args.attn_layers_num > 0:
            self.type_ids=torch.tensor(type_ids, dtype=torch.long)
            if args.add_type_embedding:
                # 3 types: node itself, neighbor nodes, anchors
                self.type_embedding = nn.Embedding(num_embeddings=3, embedding_dim=anchor_dim)
                nn.init.normal_(self.type_embedding.weight, mean=0.0, std=args.init_std)
            else:
                self.type_embedding = None
            self.attn_layers = nn.ModuleList([
                TransformerBlock(in_feat=anchor_dim, mlp_ratio=args.mlp_ratio, num_heads=4, head_dim=args.head_dim, dropout_p=args.drop, init_std=args.init_std)
                #nn.TransformerEncoderLayer(d_model=anchor_dim, nhead=4, dim_feedforward=int(anchor_dim*args.mlp_ratio), dropout=args.drop, activation='relu')
                for _ in range(args.attn_layers_num)
            ])
            self.linear=nn.Linear(anchor_dim, anchor_dim)
            nn.init.normal_(self.linear.weight, mean=0.0, std=args.init_std)
            nn.init.normal_(self.linear.bias, mean=0.0, std=args.init_std)
        del type_ids

    def _encode_MSAO(self, entities: torch.LongTensor, entity_embedding, rank) -> torch.FloatTensor:
        prev_shape = entities.shape
        if len(prev_shape) > 1:
            entities = entities.view(-1)
        
        cnt_embs = torch.index_select(
            entity_embedding,
            dim=0,
            index=entities
            ).unsqueeze(1)
        mask = torch.tensor([[1]]*entities.size(0), requires_grad=False).to(rank)

        if self.sample_anchors > 0:
            hashes = self.hashes.to(rank)[entities]
            anc_embs = torch.index_select(
                entity_embedding,
                dim=0,
                index=hashes.view(-1)
                ).view(hashes.size(0), hashes.size(1), self.kge_model.entity_dim * self.kge_model.house_dim)
            mask = torch.cat([mask, self.anc_mask.to(rank)[entities]], dim=1)
            
        embs_seq = cnt_embs
        embs_seq = torch.cat([embs_seq, anc_embs], dim=1) if self.sample_anchors > 0 else embs_seq
        
        if self.attn_layers is not None:
            if self.type_embedding is not None:
                embs_attn = embs_seq + self.type_embedding(self.type_ids.to(rank))
            for enc_layer in self.attn_layers:
                embs_attn = enc_layer(embs_attn, mask = mask)
            embs_seq = embs_seq[:,0,:] + self.dropout2(self.linear(self.dropout(embs_attn)))[:,0,:]

        return embs_seq.view(*prev_shape, -1)
    
    def _encode_cache(self, entities):
        return self.kge_model.entity_embedding[entities]

    def cache_entity_embedding(self, batch_size=1024):
        if 'entity_embedding' in self._modules:
            return
        # calculate the entity embeddings
        entity_embedding = []
        for i in range(0, self.nentity, batch_size):
            entities = torch.arange(
                i, min(i+batch_size, self.nentity), dtype=torch.long, 
                device=self.relation_embedding.weight.device)
            entity_embedding.append(self._encode_func(entities).detach())
        #self.register_buffer('entity_embedding', torch.cat(entity_embedding, dim=0))
        self.entity_embedding=torch.cat(entity_embedding, dim=0)
        self._prev_encode_func = self._encode_func
        self._encode_func = self._encode_cache
        print('cache_entity_embedding')

    def detach_entity_embedding(self):
        if 'entity_embedding' in self._modules:
            return
        assert 'entity_embedding' in self._buffers and hasattr(self, '_prev_encode_func'), \
            'The entity embeddings should be cached before detached.'
        delattr(self, 'entity_embedding')
        self._encode_func = self._prev_encode_func
        print('detach_entity_embedding')

    def tokenize_anchors(self, anchor_size=0.1, drop_size=0.05, sample_anchors=20):#data_name, nentity, all_true_triples, train_triples
        def read_previous_file(anchor_size, drop_size, sample_anchors):
            fpattern = '{}_{}_{}_{}_anchors.pkl'.format(self.data_name, anchor_size, drop_size, sample_anchors)
            for fname in no_error_listdir('data/'):
                r = re.match(fpattern, fname)
                if r:
                    print('Reading exsiting anchor file {}...'.format('data/'+fname), end='')
                    pretime = time.time()
                    anchors, masks = pickle.load(open('data/'+fname, "rb"))
                    print('{:.2f} seconds taken.'.format(time.time()-pretime))
                    return (anchors, masks)
            return None, None
        
        # create anchor file
        anchor_size = int(anchor_size if anchor_size > 1 else self.kge_model.nentity*anchor_size)
        assert anchor_size <= self.kge_model.nentity
        #drop_size = int(drop_size if anchor_size > 1 else self.kge_model.nentity*drop_size)
        #assert drop_size <= self.kge_model.nentity // 2
        anchors_vocab, masks_vocab = read_previous_file(anchor_size, drop_size, sample_anchors)
        if anchors_vocab is not None:
            return anchors_vocab, masks_vocab

        G = nx.DiGraph()
        G.add_nodes_from(range(self.kge_model.nentity))
        edges=[]
        for h,r,t in self.train_triples:
            edges.append((h,t))
        G.add_edges_from(edges)
        pagerank_list = defaultdict(float, nx.pagerank(G))
        candi_nodes = sorted(range(self.kge_model.nentity), key=lambda x: pagerank_list[x], reverse=True)[:anchor_size]
        
        ent_embs = self.kge_model.entity_embedding.view(self.kge_model.nentity + 1, self.kge_model.entity_dim * self.kge_model.house_dim)[:self.kge_model.nentity, :]
        ent_norm = (torch.sum(ent_embs**2,dim=1)**0.5).unsqueeze(1)
        ent_sim = ent_embs @ ent_embs.transpose(-2,-1) / (ent_norm @ ent_norm.transpose(-2,-1))
        ent_sim = ent_sim.masked_fill(ent_sim > 1 - drop_size, -1e15)
        ent_ranks = torch.sort(ent_sim, dim=1, descending=True).indices#[:,drop_size:self.kge_model.nentity//2]
        
        anchors_vocab, masks_vocab = [], []
        for i in range(self.kge_model.nentity):
            ranks = ent_ranks[i, :sample_anchors + 1].tolist()
            anchor = [r for r in ranks if r!=i and r in candi_nodes][:sample_anchors]
            anchors_vocab.append(anchor + [self.kge_model.nentity] * (sample_anchors - len(anchor)))
            masks_vocab.append([1]*len(anchor) + [0] * (sample_anchors - len(anchor)))
        
        anchors_vocab = torch.tensor(anchors_vocab, dtype=torch.long)
        masks_vocab = torch.tensor(masks_vocab, requires_grad=False).float()

        anchor_file = 'data/{}_{}_{}_{}_anchors.pkl'.format(self.data_name, anchor_size, drop_size, sample_anchors)
        print('Saving to {}...'.format(anchor_file), end='')
        pretime = time.time()
        pickle.dump((anchors_vocab, masks_vocab), open(anchor_file, "wb"))
        print('{:.2f} seconds taken.'.format(time.time()-pretime))

        return anchors_vocab, masks_vocab
        

    def forward(self, sample, mode='single', rank = 0, stage = '1'):
        '''
        Forward function that calculate the score of a batch of triples.
        In the 'single' mode, sample is a batch of triple.
        In the 'head-batch' or 'tail-batch' mode, sample consists two part.
        The first part is usually the positive sample.
        And the second part is the entities in the negative samples.
        Because negative samples and positive samples usually share two elements
        in their triple ((head, relation) or (relation, tail)).
        '''
        if stage == '1':
            score = self.kge_model(sample, mode)
            return score
        
        elif stage == '2':
            entity_embedding, r = self.kge_model.norm_embedding(mode)
            entity_embedding = entity_embedding.view(self.kge_model.nentity + 1, self.kge_model.entity_dim * self.kge_model.house_dim)

            if mode == 'head-batch':
                tail_part, head_part = sample
                batch_size, negative_sample_size = head_part.size(0), head_part.size(1)

                head_index = head_part.view(-1)
                relation_index = tail_part[:, 1]
                tail_index = tail_part[:, 2]
            elif mode == 'tail-batch':
                head_part, tail_part = sample
                batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)

                head_index = head_part[:, 0]
                relation_index = head_part[:, 1]
                tail_index = tail_part.view(-1)
            else:
                raise ValueError('mode %s not supported' % mode)
        
            head = self._encode_MSAO(head_index, entity_embedding, rank)
            head = head.reshape(batch_size, negative_sample_size, self.kge_model.entity_dim, self.kge_model.house_dim) if mode == 'head-batch' else head.reshape(batch_size, 1, self.kge_model.entity_dim, self.kge_model.house_dim)
            tail = self._encode_MSAO(tail_index, entity_embedding, rank)
            tail = tail.reshape(batch_size, 1, self.kge_model.entity_dim, self.kge_model.house_dim) if mode == 'head-batch' else tail.reshape(batch_size, negative_sample_size, self.kge_model.entity_dim, self.kge_model.house_dim)

            relation = torch.index_select(
                r,
                dim=0,
                index=relation_index
            ).unsqueeze(1)

            if self.kge_model.model_name == 'HousE' or self.kge_model.model_name == 'HousE_plus':
                k_head = torch.index_select(
                    self.kge_model.k_head,
                    dim=0,
                    index=relation_index
                ).unsqueeze(1)
                k_tail = torch.index_select(
                    self.kge_model.k_tail,
                    dim=0,
                    index=relation_index
                ).unsqueeze(1)

            if self.kge_model.model_name == 'HousE_r_plus' or self.kge_model.model_name == 'HousE_plus':
                re_weight = torch.index_select(
                    self.kge_model.relation_weight,
                    dim=0,
                    index=relation_index
                ).unsqueeze(1)

            if self.kge_model.model_name == 'HousE_r':
                score = self.kge_model.HousE_r(head, relation, tail, mode)
            elif self.kge_model.model_name == 'HousE':
                score = self.kge_model.HousE(head, relation, k_head, k_tail, tail, mode)
            elif self.kge_model.model_name == 'HousE_r_plus':
                score = self.kge_model.HousE_r_plus(head, relation, re_weight, tail, mode)
            elif self.kge_model.model_name == 'HousE_plus':
                score = self.kge_model.HousE_plus(head, relation, re_weight, k_head, k_tail, tail, mode)
            else:
                raise ValueError('model %s not supported' % self.kge_model.model_name)
                
            return score
        else:
            raise ValueError('stage %s not supported: ' % stage)

