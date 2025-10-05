import dgl
import numpy as np
import torch
import scipy.sparse as sp
import networkx as nx
from numpy import inf

def get_sp_adj(graph,name):
    graph =dgl.remove_self_loop(graph)
    nx_graph = dgl.to_networkx(graph)

    # 提取邻接矩阵
    adj = nx.adjacency_matrix(nx_graph)
    # 对称化邻接矩阵
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    # 添加自环
    adj = adj + sp.eye(adj.shape[0])

    # 计算度矩阵 D
    D = []
    for i in range(adj.sum(axis=1).shape[0]):
        D.append(adj.sum(axis=1)[i, 0])
    D = np.diag(D)
    l = D - adj
    with np.errstate(divide='ignore'):
        D_norm = D ** (-0.5)
    D_norm[D_norm == inf] = 0
    adj = sp.coo_matrix(D_norm.dot(l).dot(D_norm))
    sp.save_npz('Lap_matrix_{}.npz'.format(name), adj)


'''
### 稀疏的计算方式 (for large graphs) ### 
def get_sp_adj(graph, name):
    # 移除自环
    graph = dgl.remove_self_loop(graph)

    # 转换为 networkx 图
    nx_graph = dgl.to_networkx(graph)

    # 提取邻接矩阵（scipy.sparse.csr_matrix）
    adj = nx.adjacency_matrix(nx_graph)

    # 对称化邻接矩阵
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    # 添加自环
    adj = adj + sp.eye(adj.shape[0])

    # 计算度向量
    degrees = np.array(adj.sum(axis=1)).flatten()

    # 构造 D^(-0.5) 稀疏对角矩阵
    with np.errstate(divide='ignore'):
        deg_inv_sqrt = np.power(degrees, -0.5)
        deg_inv_sqrt[np.isinf(deg_inv_sqrt)] = 0.0
    D_inv_sqrt = sp.diags(deg_inv_sqrt)

    # 计算规范化邻接矩阵：I-D^{-0.5} * A * D^{-0.5}
    norm_adj = sp.eye(adj.shape[0]) - D_inv_sqrt @ adj @ D_inv_sqrt

    # 转为 COO 格式并保存
    norm_adj = norm_adj.tocoo()
    sp.save_npz(f'adj_matrix_{name}.npz', norm_adj)
'''
# 加载图并调用函数
graph = dgl.load_graphs('dataset/name')[0][0]
graph = dgl.to_simple(graph) 
get_sp_adj(graph, 'name')
