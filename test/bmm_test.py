import pygho
import pygho.backend.Spspmm as spspmm
import numpy as np
import torch
import pygho.backend.Mamamm as mamamm
from torch_geometric.utils import k_hop_subgraph

def sparse_bmm(ei1, x1, ei2, x2, n):
    x1 = pygho.SparseTensor(ei1, x1, (n, n, x1.shape[1]))
    x2 = pygho.SparseTensor(ei2, x2, (n, n, x2.shape[1]))
    res = spspmm.spspmm(x1, 1, x2, 0, "sum")
    return res.indices, res.values

def combine(x, v, xx, pos):
    #print(x.shape, v.shape, xx.shape, pos.shape)
    pos = pos.to('cpu').numpy()
    x = x.to('cpu').numpy()
    dtype = [('u', int), ('v', int)]
    x = np.core.records.fromarrays(x, dtype=dtype)
    pos = np.core.records.fromarrays(pos, dtype=dtype)
    _, pos_indice, x_indice = np.intersect1d(pos, x, return_indices=True) 
    pos_val = torch.zeros((pos.shape[0], v.shape[1]), device=v.device)
    pos_val[pos_indice] = v[x_indice]
    return torch.cat([pos_val, xx], dim=1)

def subgraph_extraction(pos, ei, x):
    pos_nodes = torch.cat([torch.unique(pos[0]), torch.unique(pos[1])], dim=0)
    subset, ei, new_pos_nodes, _ = k_hop_subgraph(pos_nodes, 1, ei, relabel_nodes=True)
    mapping = torch.full((pos_nodes.max().item() + 1, ), -1, dtype=torch.long, device=pos_nodes.device)
    mapping[pos_nodes] = new_pos_nodes
    pos = mapping[pos]
    x = x[subset]
    return subset, pos, ei, x
    
def test_subgraph_extraction():
    pos = torch.tensor([[2], [3]], dtype=torch.long)

    ei = torch.tensor([[0, 1, 2, 3, 1, 2, 3, 4], [1, 2, 3, 4, 0, 1, 2, 3]], dtype=torch.long)

    x = torch.arange(5, dtype=torch.float).view(-1, 1)

    subset, pos, ei, x = subgraph_extraction(pos, ei, x)

    print("subset:", subset)
    print("pos:", pos)
    print("ei:", ei)
    print("x:", x)


def dense_bmm2(ei1, x1, ei2, x2, n):
    adj = torch.zeros((n, n, x1.shape[1]), dtype = torch.float32)
    adj[ei1[0], ei1[1]] = x1
    x1 = adj.permute(2, 0, 1)
    adj = torch.zeros((n, n, x2.shape[1]), dtype = torch.float32)
    adj[ei2[0], ei2[1]] = x2
    x2 = adj.permute(2, 0, 1)
    res = torch.bmm(x1, x2)
    res = res.permute(1, 2, 0)
    mask = torch.any(res != 0, dim=-1)
    mask = torch.nonzero(mask)
    return mask.t(), res[mask[:, 0], mask[:, 1], :]


def test_dense_bmm():
    n = 5
    
    ei1 = torch.tensor([[0, 2, 3], [1, 4, 2]], dtype=torch.long)
    x1 = torch.tensor([[2.0, 3.0], [4.0, 5.0], [6.0, 7.0]], dtype=torch.float32)
    
    ei2 = torch.tensor([[0, 1, 4], [2, 3, 3]], dtype=torch.long)
    x2 = torch.tensor([[8.0, 9.0], [10.0, 11.0], [12.0, 13.0]], dtype=torch.float32)

    indices, values = dense_bmm2(ei1, x1, ei2, x2, n)

    print("indices:", indices)
    print("values:", values)

def test_sparse_bmm():
    n = 5
    
    ei1 = torch.tensor([[0, 2, 3], [1, 4, 2]], dtype=torch.long)
    x1 = torch.tensor([[2.0, 3.0], [4.0, 5.0], [6.0, 7.0]], dtype=torch.float32)
    
    ei2 = torch.tensor([[0, 1, 4], [2, 3, 3]], dtype=torch.long)
    x2 = torch.tensor([[8.0, 9.0], [10.0, 11.0], [12.0, 13.0]], dtype=torch.float32)

    indices, values = dense_bmm2(ei1, x1, ei2, x2, n)

    print("indices:", indices)
    print("values:", values)



def test_combine():
    # Sample input data
    x = torch.tensor([[2, 1], [1, 3], [0, 2]], dtype=torch.long).t()
    v = torch.tensor([[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]], dtype=torch.float32)
    xx = torch.tensor([[11.0, 12.0], [13.0, 14.0], [15.0, 16.0]], dtype=torch.float32)
    pos = torch.tensor([[1, 0], [2, 3], [2, 1]], dtype=torch.long).t()

    # Run the combine function
    output = combine(x, v, xx, pos)

    # Check if the output matches the expected output
    print("output:", output)

test_sparse_bmm()
test_dense_bmm()
test_combine()
test_subgraph_extraction()