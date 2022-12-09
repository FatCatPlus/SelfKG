# coding: UTF-8
import argparse
import collections
import logging
import os
import pickle
import random
import sys
from datetime import datetime
from posixpath import join

sys.path.append('..')
import faiss
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# using labse
# from transformers import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
from torch.nn import *

from loader.DBP15kRawLoader import DBP15kRawLoader, ENDBP15kRawLoader
from loader.DBP15KRawNeighbors import DBP15KRawNeighbors
from model.layers_LaBSE_neighbor import MyEmbedder, Trainer
from script.preprocess.deal_raw_dataset import MyRawdataset
from settings import *

sys.argv=['']
del sys
# Labse embedding dim
MAX_LEN = 88

LANGUAGE = 'zh_en'
EPOCH = 124
PLT_LAN = 'ZH'

def parse_options(parser):
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--time', type=str, default=datetime.now().strftime("%Y%m%d%H%M%S"))
    parser.add_argument('--language', type=str, default=LANGUAGE)
    parser.add_argument('--model_language', type=str, default=LANGUAGE)
    parser.add_argument('--model', type=str, default='LaBSE')

    parser.add_argument('--epoch', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--queue_length', type=int, default=64)

    parser.add_argument('--center_norm', type=bool, default=False)
    parser.add_argument('--neighbor_norm', type=bool, default=True)
    parser.add_argument('--emb_norm', type=bool, default=True)
    parser.add_argument('--combine', type=bool, default=True)

    parser.add_argument('--gat_num', type=int, default=1)

    parser.add_argument('--t', type=float, default=0.08)
    parser.add_argument('--momentum', type=float, default=0.9999)
    parser.add_argument('--lr', type=float, default=1e-6)
    parser.add_argument('--dropout', type=float, default=0.3)

    return parser.parse_args()

def neighbor_dict(language, id_entity, num=1 ):
    id_neighbors_dict = {}
    filepath = PROJ_DIR + f'/data/DBP15K/{language}/triples_{num}'
    data = pd.read_csv(filepath, header=None, sep='\t')
    data.columns = ['head', 'relation', 'tail']

    for index, row in data.iterrows():
        head_str = id_entity[int(row['head'])]
        tail_str = id_entity[int(row['tail'])]
        # print(head_str)
        if not id_entity[int(row['head'])] in id_neighbors_dict.keys():
            id_neighbors_dict[id_entity[int(row['head'])]] = [head_str]
        if not tail_str in id_neighbors_dict[id_entity[int(row['head'])]]:
            id_neighbors_dict[id_entity[int(row['head'])]].append(tail_str)
        
        if not id_entity[int(row['tail'])] in id_neighbors_dict.keys():
            id_neighbors_dict[id_entity[int(row['tail'])]] = [tail_str]
        if not head_str in id_neighbors_dict[id_entity[int(row['tail'])]]:
            id_neighbors_dict[id_entity[int(row['tail'])]].append(head_str)
    return id_neighbors_dict

def gen(id_entity_1, id_entity_2, id_neighbors_dict1, id_neighbors_dict2, fail_ids2):
    _list = []
    _num = []
    for _id2 in fail_ids2:
        _id1 = link[_id2]
        tempdict = {}
        tempdict[f'{LANGUAGE[:2]}_entity'] = id_entity_1[_id1]
        tempdict[f'{LANGUAGE[:2]}_entity_neighbors'] = id_neighbors_dict1[id_entity_1[_id1]][1:]
        tempdict[f'{LANGUAGE[:2]}_entity_num'] = str(len(id_neighbors_dict1[id_entity_1[_id1]])-1)
        tempdict['en_entity'] = id_entity_2[_id2]
        tempdict['en_entity_neighbors'] = id_neighbors_dict2[id_entity_2[_id2]][1:]
        tempdict['en_entity_num'] = str(len(id_neighbors_dict2[id_entity_2[_id2]])-1)
        _num.append([len(id_neighbors_dict1[id_entity_1[_id1]])-1, len(id_neighbors_dict2[id_entity_2[_id2]])-1])
        _list.append(tempdict)
    return _list, _num

def writer(language, hit, _list):
    with open(FAIL_DIR + f'/{language}_failed_{hit}.json', 'w', encoding='utf-8') as f:
        for my_dict in _list:
            for item in my_dict.items():
                json.dump(item, f, ensure_ascii=False)
                f.write('\n')
                if item[0] == 'en_entity_num':
                    f.write('\n')

parser = argparse.ArgumentParser()
args = parse_options(parser)

path = PROJ_DIR + f'/checkpoints/LaBSE/{args.language}/model_neighbor_True_epoch_{EPOCH}_batch_size_96_neg_queue_len_63.ckpt'

model = MyEmbedder(args, VOCAB_SIZE)
checkpoint = torch.load(path)
model.load_state_dict(checkpoint['model'])

evaluator = Trainer(args, training=False)
evaluator.model = model.cuda()
_1, _2, _3, _4, fail_ids2_at1, fail_ids2_at10 = evaluator.evaluate(EPOCH)

link = evaluator.link

id_entity_1 = DBP15kRawLoader(language=args.language).id_entity
id_entity_2 = ENDBP15kRawLoader(language=args.language).id_entity
id_neighbors_dict1 = neighbor_dict(args.language, id_entity_1, num=1)
id_neighbors_dict2 = neighbor_dict(args.language, id_entity_2, num=2)
at1_list, at1_num = gen(id_entity_1, id_entity_2, id_neighbors_dict1, id_neighbors_dict2, fail_ids2_at1)
at10_list, at10_num = gen(id_entity_1, id_entity_2, id_neighbors_dict1, id_neighbors_dict2, fail_ids2_at10)

writer(LANGUAGE, 'hit1', at1_list)
writer(LANGUAGE, 'hit10', at10_list)

with open(FAIL_DIR + f'/{LANGUAGE}_num.pkl', 'wb') as f:
    pickle.dump([at1_num, at10_num], f)

plt.figure()
ax1 = plt.subplot(2,2,1)
n, bins, patches = plt.hist(x=np.array(at1_num)[:,0], bins=20, range=(0,100), color='#0504aa',
alpha=0.7, rwidth=0.85)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Entity Degree')
plt.ylabel('Frequency')
plt.title(f'{PLT_LAN} Failed Case Hit1')
maxfreq = n.max()
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq+10)


ax2 = plt.subplot(2,2,2)
n, bins, patches = plt.hist(x=np.array(at1_num)[:,1], bins=20, range=(0,100), color='#0504aa',
alpha=0.7, rwidth=0.85)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Entity Degree')
plt.ylabel('Frequency')
plt.title('EN Failed Case Hit1')
maxfreq = n.max()
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq+10)

ax3 = plt.subplot(2,2,3)
n, bins, patches = plt.hist(x=np.array(at10_num)[:,0], bins=20, range=(0,100), color='#0504aa',
alpha=0.7, rwidth=0.85)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Entity Degree')
plt.ylabel('Frequency')
plt.title(f'{PLT_LAN} Failed Case Hit10')
maxfreq = n.max()
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq+10)

ax4 = plt.subplot(2,2,4)
n, bins, patches = plt.hist(x=np.array(at10_num)[:,1], bins=20, range=(0,100), color='#0504aa',
alpha=0.7, rwidth=0.85)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Entity Degree')
plt.ylabel('Frequency')
plt.title('EN Failed Case Hit10')
maxfreq = n.max()
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq+10)

plt.tight_layout()
plt.savefig(IMG_DIR +f'/{LANGUAGE}_longtail_failed.png')
# plt.show()