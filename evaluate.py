# coding: UTF-8
import argparse
import logging
import os
import random
import sys
from datetime import datetime
from posixpath import join

import faiss
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

from loader.DBP15KRawNeighbors import DBP15KRawNeighbors
from model.layers_LaBSE_neighbor import *
from script.preprocess.deal_raw_dataset import MyRawdataset
from settings import *

sys.argv=['']
del sys
# Labse embedding dim
MAX_LEN = 88

path = '/home1/data5/bowen/code/SelfKG/checkpoints/LaBSE/zh_en/model_neighbor_True_epoch_124_batch_size_96_neg_queue_len_63.ckpt'

parser = argparse.ArgumentParser()
args = parse_options(parser)
model = MyEmbedder(args, VOCAB_SIZE)
checkpoint = torch.load(path)
model.load_state_dict(checkpoint['model'])
epoch = checkpoint['epoch']

evaluator = Trainer(training=False)
evaluator.model = model.cuda()
_1, _2, _3, _4, fail_ids2_at1 = evaluator.evaluate(124)
# print(fail_ids2_at1)
triple_1 = DBP15KRawNeighbors.id_neighbors_loader




# with open(FAIL_DIR + 'hit1_failed_2.txt','wb') as f:
#     for line in fail_ids_at1:
#         line = str(line)
#         a = line.strip().split("\t")
#         f.write(a+'\n')