
import scipy
import anndata as ad
import scanpy as sc
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from scipy.sparse import issparse, csr
from anndata import AnnData
from sklearn.preprocessing import maxabs_scale, MaxAbsScaler

CHUNK_SIZE = 20000


def reindex(adata, genes, chunk_size=CHUNK_SIZE):
    """
    Reindex AnnData with gene list
    
    Parameters
    ----------
    adata
        AnnData
    genes
        gene list for indexing
    chunk_size
        chunk large data into small chunks
        
    Return
    ------
    AnnData
    """
    idx = [i for i, g in enumerate(genes) if g in adata.var_names]
    print('There are {} gene in selected genes'.format(len(idx)))
    if len(idx) == len(genes):
        adata = adata[:, genes]
    else:
        new_X = scipy.sparse.lil_matrix((adata.shape[0], len(genes)))
        for i in range(new_X.shape[0]//chunk_size+1):
            new_X[i*chunk_size:(i+1)*chunk_size, idx] = adata[i*chunk_size:(i+1)*chunk_size, genes[idx]].X
        adata = AnnData(new_X.tocsr(), obs=adata.obs, var={'var_names':genes}) 
    return adata

def plot_hvg_umap(hvg_adata,color=['celltype'],path = None, save_filename=None):
    sc.set_figure_params(dpi=80, figsize=(3,3)) # type: ignore
    hvg_adata = hvg_adata.copy()
    if save_filename:
        sc.settings.figdir = path
        # save = '.pdf'
        save = f'{save_filename}.pdf'
    else:
        save = None
    # ideal gas equation
    
    sc.pp.scale(hvg_adata, max_value=10)
    sc.tl.pca(hvg_adata)
    sc.pp.neighbors(hvg_adata, n_pcs=30, n_neighbors=30)
    sc.tl.umap(hvg_adata, min_dist=0.1)
    sc.pl.umap(hvg_adata, color=color,legend_fontsize=15, ncols=2 ,show=None,save=save)
    return hvg_adata


# def get_data_loader(data_ary:np.ndarray, 
#                     cell_type:np.ndarray, 
#                     LC,
#                     batch_size:int=512,
#                     is_shuffle:bool=True,
#                     ):
    
#         data_tensor = torch.from_numpy(data_ary.astype(np.float32))
#         cell_type_tensor = torch.from_numpy(cell_type.astype(np.float32))
#         LC=torch.tensor(LC).long()
#         dataset = TensorDataset(data_tensor,cell_type_tensor,LC)
#         generator = torch.Generator(device='cuda')
#         return DataLoader(
#                 dataset, batch_size=batch_size, shuffle=is_shuffle, drop_last=False , generator=generator) #, generator=torch.Generator(device = 'cuda') 
          

        
def get_data_loader(data_ary: np.ndarray, 
                    cell_type: np.ndarray, 
                    LC,
                    batch_size: int = 512,
                    is_shuffle: bool = True,
                    worker_init_fn = None 
                   ):
    """
    功能：
        将 numpy 数据（基因表达矩阵、细胞类型、空间坐标等）转换成 PyTorch 可用的 DataLoader，以便后续输入模型训练或推理。
    参数：
        data_ary (np.ndarray): 主输入数据（通常是细胞或spot的基因表达矩阵），形状如 (n_cells, n_genes)
        cell_type (np.ndarray): 细胞类型标签或辅助特征，形状一般为 (n_cells, n_types) 或 (n_cells,)
        LC : 空间坐标信息（二维数组，形如 (n_cells, 2)）或者其他位置信息矩阵
        batch_size (int): 每个批次的数据量，默认为 512
        is_shuffle (bool): 是否在每个 epoch 打乱数据顺序，默认为 True
    返回：torch.utils.data.DataLoader 对象用于在训练时按批次加载数据
    """

    # 将 numpy 数组转成 PyTorch 的 float32 张量
    data_tensor = torch.from_numpy(data_ary.astype(np.float32))

    # 将 cell_type 转成 float32 张量（可作为回归标签或特征输入）
    cell_type_tensor = torch.from_numpy(cell_type.astype(np.float32))

    # 将 LC 转换为 long 类型张量（通常用于分类或索引）
    LC = torch.tensor(LC, dtype=torch.float32)

    # 将三个张量打包成 TensorDataset，每个样本包含 (data, cell_type, LC)
    dataset = TensorDataset(data_tensor, cell_type_tensor, LC)

    # 创建随机数生成器，指定运行设备为 GPU
    generator = torch.Generator(device='cuda')

    # 构建 DataLoader：可按批次加载数据并支持随机打乱
    return DataLoader(
        dataset, 
        batch_size=batch_size,   # 每批加载样本数
        shuffle=is_shuffle,      # 是否打乱数据
        drop_last=False,         # 是否丢弃最后不足一个 batch 的数据
        generator=generator,
        worker_init_fn=worker_init_fn,    # 使用 GPU 上的随机数生成器
    )        

    
# def get_data_loader(data_ary: np.ndarray,
#                     cell_type: np.ndarray,
#                     batch_size: int = 512,
#                     is_shuffle: bool = True,
#                     ):
#     data_tensor = torch.from_numpy(data_ary.astype(np.float32))
#     cell_type_tensor = torch.from_numpy(cell_type.astype(np.float32))
#     dataset = TensorDataset(data_tensor, cell_type_tensor)
#     generator = torch.Generator(device='cuda')
#     return DataLoader(
#         dataset, batch_size=batch_size, shuffle=is_shuffle, drop_last=False,
#         generator=generator)  # , generator=torch.Generator(device = 'cuda')    
    


def scale(adata):
    scaler = MaxAbsScaler()
    normalized_data = scaler.fit_transform(adata.X.T).T
    adata.X = normalized_data
    return adata


def data_augment(adata: AnnData, fixed: bool, noise_std):
   
    noise_stddev = noise_std
    augmented_adata = adata.copy()
    gene_expression = adata.X
    
    if fixed: 
        augmented_adata.X = augmented_adata.X + np.full(gene_expression.shape, noise_stddev)
    else:
        augmented_adata.X = augmented_adata.X + np.abs(np.random.normal(0, noise_stddev, gene_expression.shape))   
    
    merge_adata = adata.concatenate(augmented_adata, join='outer')
    
    
    return merge_adata
    

    
    
