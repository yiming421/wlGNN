import torch
import pygho.backend.Spspmm as spspmm
import pygho
import time
from torch_geometric.utils import to_undirected

def dense_bmm(ei1, x1, ei2, x2, n):
    adj = torch.zeros((n, n, x1.shape[1]), dtype = torch.float32, device = x1.device)
    adj[ei1[0], ei1[1]] = x1
    x1 = adj.permute(2, 0, 1)
    adj = torch.zeros((n, n, x2.shape[1]), dtype = torch.float32, device = x1.device)
    adj[ei2[0], ei2[1]] = x2
    x2 = adj.permute(2, 0, 1)
    res = torch.bmm(x1, x2)
    res = res.permute(1, 2, 0)
    mask = torch.any(res != 0, dim=-1)
    mask = torch.nonzero(mask)
    return mask.t(), res[mask[:, 0], mask[:, 1], :]

def test_1():
    n, m = 3, 10000000
    ei1 = torch.tensor([[0, 2], [1, 2]], dtype=torch.long).cuda()
    x1 = torch.randn(2).cuda()
    ei2 = torch.tensor([[0, 1], [2, 2]], dtype=torch.long).cuda()
    x2 = torch.randn(2).cuda()
    st = time.time()
    print(ei1.shape, x1.shape)
    print(ei2.shape, x2.shape)
    x11 = pygho.SparseTensor(ei1, x1, (n, n))
    x21 = pygho.SparseTensor(ei2, x2, (n, n))
    res = spspmm.spspmm(x11, 1, x21, 0, "sum")
    en = time.time()
    print(f"Time: {en - st}")
    print(f"memory: {torch.cuda.max_memory_allocated() / 1024 / 1024} MB")

    st = time.time()
    x12 = pygho.SparseTensor(ei1, x1, (m, m))
    x22 = pygho.SparseTensor(ei2, x2, (m, m))
    res = spspmm.spspmm(x12, 1, x22, 0, "sum")
    en = time.time()
    print(f"Time: {en - st}")
    print(f"memory: {torch.cuda.max_memory_allocated() / 1024 / 1024} MB")

def test_2():
    n = 20000
    dense_ratio = 0.01
    e = 200000

    ei1 = torch.randint(0, n, (2, e), dtype=torch.long).cuda()
    ei2 = torch.randint(0, n, (2, e), dtype=torch.long).cuda()
    x1 = torch.randn(e, 2).cuda()
    x2 = torch.randn(e, 2).cuda()
    st = time.time()
    #res = dense_bmm(ei1, x1, ei2, x2, n)
    x11 = pygho.SparseTensor(ei1, x1, (n, n, 2))
    x21 = pygho.SparseTensor(ei2, x2, (n, n, 2))
    res = spspmm.spspmm(x11, 1, x21, 0, "sum")
    en = time.time()
    print(f"Time: {en - st}")
    print(f"memory: {torch.cuda.max_memory_allocated() / 1024 / 1024} MB")

def test_3():
    ei = torch.tensor([[0, 1, 2, 3, 1, 2, 3, 4, 3, 1], [1, 2, 3, 4, 0, 1, 2, 3, 1, 3]], dtype=torch.long).cuda()
    x = torch.arange(5, dtype=torch.float).view(-1, 1).cuda()
    pos = torch.tensor([[2], [3]], dtype=torch.long).cuda()
    mask = ~((ei[0] == pos[0]) & (ei[1] == pos[1]))
    print(mask.shape, flush = True)
    ei = ei[:, mask]
    print(ei, flush = True)
    mask = torch.randint(0, ei.size(1), (4,))
    print(mask, flush = True)
    ei = ei[:, mask]
    print(ei, flush = True)
    ei = to_undirected(ei)
    print(ei, flush = True)
    ei = torch.cat([ei, pos], dim=1)
    print(ei, flush = True)

test_2()

