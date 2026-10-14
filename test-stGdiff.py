#!/usr/bin/env python
# coding: utf-8

# ## https://github.com/fdu-wangfeilab/stGDiff

import os
import argparse
import csv
import itertools
import time
import warnings

import numpy as np
import pandas as pd
import scanpy as sc
import torch
import scipy.stats as st

from process.data import *

warnings.filterwarnings('ignore')
torch.set_default_tensor_type('torch.cuda.FloatTensor')

n_splits = 5

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument("--sc_data", type=str, default="dataset8_seq_84.h5ad")
parser.add_argument("--sp_data", type=str, default='dataset8_spatial_84.h5ad')
parser.add_argument("--document", type=str, default='dataset8')
parser.add_argument("--batch_size", type=int, default=2048)
parser.add_argument("--hidden_size", type=int, default=512)
parser.add_argument("--noise_std", type=float, default=10)
parser.add_argument("--head", type=int, default=16)
parser.add_argument("--step", type=int, default=1500)
parser.add_argument("--epoch", type=int, default=900)
parser.add_argument("--rand", type=int, default=0)
args = parser.parse_known_args()[0]


# 数据处理
sc.settings.seed = 42
adata_spatial = sc.read_h5ad('/root/autodl-fs/sp/' + args.sp_data)
print(adata_spatial.shape)
adata_seq = sc.read_h5ad('/root/autodl-fs/sc/' + args.sc_data)
print(adata_seq.shape)
adata_seq2 = data_augment(adata_seq.copy(), True, noise_std=args.noise_std)
adata_spatial2 = adata_spatial.copy()

sc.pp.normalize_total(adata_seq2, target_sum=1e4)
sc.pp.log1p(adata_seq2)
adata_seq2 = scale(adata_seq2)

sc.pp.pca(adata_seq2)
sc.pp.neighbors(adata_seq2)
sc.tl.umap(adata_seq2)

data_seq_array = adata_seq2.X
print(data_seq_array.shape)

sc.pp.normalize_total(adata_spatial2, target_sum=1e4)
sc.pp.log1p(adata_spatial2)
adata_spatial2 = scale(adata_spatial2)

data_spatial_array = adata_spatial2.X

adata_spatial2.obsm["spatial"] = np.column_stack((adata_spatial2.obs['x_coord'], adata_spatial2.obs['y_coord'])).astype(np.float32)

sp_genes = np.array(adata_spatial.var_names)
sp_data = pd.DataFrame(data=data_spatial_array, columns=sp_genes)
sc_data = pd.DataFrame(data=data_seq_array, columns=sp_genes)


# 跑程序
import tool.file_tool as ft
import main

this_time = int(time.time())
date = time.strftime("%m%d", time.localtime()) + "_" + str(this_time)

Data = args.document
num_epoch = args.epoch
diffusion_step = args.step
batch_size = args.batch_size
hidden_size = args.hidden_size
head = args.head
random_state = args.rand

dataset_name = Data
cluster_method = 'leiden'
seed_list = np.array(range(1202, 2020))

idix = 2000

path1 = '/root/stGdiff/data/' + Data + '/' + str(date)

date = str(date)

if not os.path.exists(path1):
    ft.mkdir(path1)


for seed in seed_list:

    impu, medians = main.run(
        idix=idix, seed=seed, path1=path1,
        adata_seq2=adata_seq2, adata_spatial2=adata_spatial2,
        sp_genes=sp_genes, data_seq_array=data_seq_array,
        Data=Data, num_epoch=num_epoch, diffusion_step=diffusion_step,
        batch_size=batch_size, hidden_size=hidden_size, head=head,
        adata_spatial=adata_spatial, n_splits=n_splits,
        random_state=random_state, data_spatial_array=data_spatial_array, date=date
    )

    index = medians.index
    medians = medians.reset_index(drop=True)
    print("---------4---------", medians)
    medians = medians.to_dict('index')
    print("---------5---------", medians)

    print("=====record parameters and results=====")

    metrics_range = list(medians.keys())
    print('metrics_range', metrics_range)
    metrics_name_use = ['SPCC', 'SSIM', 'RMSE', 'JS', "ARI", "AMI", "Homo", "NMI"]

    metrics_range_list = [[m] * len(metrics_name_use) for m in metrics_range]
    metrics_range_list = list(itertools.chain.from_iterable(metrics_range_list))
    metrics_name_list = metrics_name_use * len(metrics_range)

    print("metrics_range_list:", metrics_range_list)

    loge_len = len(metrics_range_list) * len(metrics_name_list)

    metrics_value_list = []
    for i in metrics_range:
        for j in metrics_name_use:
            if j in medians[i]:
                metrics_value_list.append(medians[i][j])
            else:
                print(f"Key '{j}' not found in medians[{i}]. Available keys: {list(medians[i].keys())}")

    dataset_name_list = [dataset_name] * loge_len
    cluster_method_list = [cluster_method] * loge_len
    current_seed_list = [seed] * loge_len

    data_block = zip(
        dataset_name_list, cluster_method_list, current_seed_list,
        metrics_range_list, metrics_name_list, metrics_value_list,
    )

    if os.path.exists('{}/result-{}.csv'.format(path1, date)):
        with open('{}/result-{}.csv'.format(path1, date), 'a', newline='') as f:
            writer = csv.writer(f)
            for item in data_block:
                writer.writerow(item)
            print("结果写入成功...")
    else:
        with open('{}/result-{}.csv'.format(path1, date), 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["dataset", "cluster_name", 'seed', 'metrics_range', "metrics_name", "metrics_value"])
            for item in data_block:
                writer.writerow(item)
            print("表头创建成功...")
    idix += 1
