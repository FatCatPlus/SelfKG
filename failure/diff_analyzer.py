import collections
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append('..')
from loader.DBP15kRawLoader import DBP15kRawLoader, ENDBP15kRawLoader
from loader.DBP15KRawNeighbors import DBP15KRawNeighbors
from model.layers_LaBSE_neighbor import MyEmbedder, Trainer
from script.preprocess.deal_raw_dataset import MyRawdataset
from settings import *

LANGUAGE = 'zh_en'

def neighbor_dict(language, id_entity, num=1):
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


link = {}
f = 'test.ref'
link_data = pd.read_csv(join(join(DATA_DIR, 'DBP15K', LANGUAGE), f), sep='\t', header=None)
link_data.columns = ['entity1', 'entity2']
entity1_id = link_data['entity1'].values.tolist()
entity2_id = link_data['entity2'].values.tolist()
for i, _ in enumerate(entity1_id):
    link[entity1_id[i]] = entity2_id[i]
    link[entity2_id[i]] = entity1_id[i]

id_entity_1 = DBP15kRawLoader(language=LANGUAGE).id_entity
id_entity_2 = ENDBP15kRawLoader(language=LANGUAGE).id_entity
id_neighbors_dict1 = neighbor_dict(LANGUAGE, id_entity_1, num=1)
id_neighbors_dict2 = neighbor_dict(LANGUAGE, id_entity_2, num=2)

_num = []
for _id2 in entity2_id:
    _id1 = link[_id2]
    _num.append([len(id_neighbors_dict1[id_entity_1[_id1]])-1, len(id_neighbors_dict2[id_entity_2[_id2]])-1])

def right_percent(_num):
    with open(FAIL_DIR + f'/{LANGUAGE}_num.pkl', 'rb') as f:
        at1_num, at10_num = pickle.load(f)

    language_all_num = np.array(_num)[:,0]
    en_all_num = np.array(_num)[:,1]
    language_at1_num = np.array(at1_num)[:,0]
    en_at1_num = np.array(at1_num)[:,1]
    language_at10_num = np.array(at10_num)[:,0]
    en_at10_num = np.array(at10_num)[:,1]

    language_diff_at1 = np.copy(language_all_num).tolist()
    for i in range(len(language_at1_num)):
        for item in language_diff_at1:
            if language_at1_num[i] == item:
                language_diff_at1.remove(item)
                break
    language_diff_at10 = np.copy(language_all_num).tolist()
    for i in range(len(language_at10_num)):
        for item in language_diff_at10:
            if language_at10_num[i] == item:
                language_diff_at10.remove(item)
                break
    en_diff_at1 = np.copy(en_all_num).tolist()
    for i in range(len(en_at1_num)):
        for item in en_diff_at1:
            if en_at1_num[i] == item:
                en_diff_at1.remove(item)
                break
    en_diff_at10 = np.copy(en_all_num).tolist()
    for i in range(len(en_at10_num)):
        for item in en_diff_at10:
            if en_at10_num[i] == item:
                en_diff_at10.remove(item)
                break
    language_at1_df = pd.DataFrame(language_diff_at1)
    language_at1_bins = [i for i in range(0,200) if i%5==0 ]
    language_at1_ = pd.cut(language_at1_df.values.flatten(), bins=language_at1_bins)

    language_at10_df = pd.DataFrame(language_diff_at10)
    language_at10_bins = [i for i in range(0,200) if i%5==0 ]
    language_at10_ = pd.cut(language_at10_df.values.flatten(), bins=language_at10_bins)

    language_all_df = pd.DataFrame(language_all_num)
    language_all_bins = [i for i in range(0,200) if i%5==0 ]
    language_all_ = pd.cut(language_all_df.values.flatten(), bins=language_all_bins)
    
    at1_df = pd.DataFrame(en_diff_at1)
    at1_bins = [i for i in range(0,200) if i%5==0 ]
    at1_ = pd.cut(at1_df.values.flatten(), bins=at1_bins)

    at10_df = pd.DataFrame(en_diff_at10)
    at10_bins = [i for i in range(0,200) if i%5==0 ]
    at10_ = pd.cut(at10_df.values.flatten(), bins=at10_bins)

    all_df = pd.DataFrame(en_all_num)
    all_bins = [i for i in range(0,200) if i%5==0 ]
    all_ = pd.cut(all_df.values.flatten(), bins=all_bins)

    a = pd.Series(language_at1_.value_counts().values / (language_all_.value_counts().values+1))
    b = pd.Series(language_at10_.value_counts().values / (language_all_.value_counts().values+1))
    c = pd.Series(at1_.value_counts().values / (all_.value_counts().values+1))
    d = pd.Series(at10_.value_counts().values / (all_.value_counts().values+1))
    return a,b,c,d

a,b,c,d = right_percent(_num)

plt.figure()

ax1 = plt.subplot(2,2,1)
plt.plot(a)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Entity Degree')
plt.ylabel('Frequency')
plt.title(f'{str.upper(LANGUAGE)[:2]} Predicted Percentage at1')

ax2 = plt.subplot(2,2,2)
plt.plot(c)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Entity Degree')
plt.ylabel('Frequency')
plt.title(f'{str.upper(LANGUAGE)[-2:]} Predicted Percentage at1')

ax3 = plt.subplot(2,2,3)
plt.plot(b)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Entity Degree')
plt.ylabel('Frequency')
plt.title(f'{str.upper(LANGUAGE)[:2]} Predicted Percentage at10')

ax4 = plt.subplot(2,2,4)
plt.plot(d)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Entity Degree')
plt.ylabel('Frequency')
plt.title(f'{str.upper(LANGUAGE)[-2:]} Predicted Percentage at10')

plt.tight_layout()
plt.savefig(IMG_DIR +f'/{LANGUAGE}_predicted_percentage.png')