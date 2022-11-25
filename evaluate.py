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

with torch.no_grad():
    model.eval()
    for 