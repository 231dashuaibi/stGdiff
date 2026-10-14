import numpy as np
import scanpy as sc
import os 

import torch
import random

# def set_seed(seed):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.benchmark=False
#     torch.backends.cudnn.deterministic=True
#     os.environ['PYTHONHASHSEED'] = str(seed)
#     固定DataLoader多进程随机种子
#     def worker_init_fn(worker_id):
#         worker_seed = seed + worker_id
#         random.seed(worker_seed)
#         np.random.seed(worker_seed)
    
#     # 🔥 适配你的调用：只返回worker_init_fn（不改你原有run函数代码）
#     return worker_init_fn


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # 🔥 新增：固定DataLoader多进程随机种子（必加！）
    def _worker_init_fn(worker_id):
        worker_seed = seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
    global worker_init_fn
    worker_init_fn = _worker_init_fn
    
    
# def seed_everything(seed: int):
#     random.seed(seed)
#     os.environ['PYTHONHASHSEED'] = str(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False