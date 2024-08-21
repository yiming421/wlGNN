import argparse
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GraphNorm
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
import itertools
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
import numpy as np
from torch_geometric.utils import negative_sampling
from torch_geometric.utils import k_hop_subgraph, add_self_loops, to_undirected
import torch_sparse
import time
import pygho
import pygho.backend.Spspmm as spspmm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dp', type=float, default=0.2)
    parser.add_argument('--dataset', type=str, default='ogbl-collab')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--hidden', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batchsize', type=int, default=768)
    parser.add_argument('--step_lr_decay', action='store_true', default=True)
    parser.add_argument('--interval', type=int, default=50)
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--metric', type=str, default='hits@50')
    parser.add_argument('--edge_threhold', type=int, default=1000000)
    parser.add_argument('--dense_ratio', type=float, default=0.01)
    return parser.parse_args()


args = parse_args()
print(args)

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

def sparse_bmm(ei1, x1, ei2, x2, n):
    x1 = pygho.SparseTensor(ei1, x1, (n, n, x1.shape[1]))
    x2 = pygho.SparseTensor(ei2, x2, (n, n, x2.shape[1]))
    #print(ei1.shape, x1.shape, ei2.shape, x2.shape, flush = True)
    res = spspmm.spspmm(x1, 1, x2, 0, "sum")
    return res.indices, res.values

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


def combine(x, v, xx, pos):
    pos = pos.to('cpu').numpy()
    x = x.to('cpu').numpy()
    dtype = [('u', int), ('v', int)]
    x = np.core.records.fromarrays(x, dtype=dtype)
    pos = np.core.records.fromarrays(pos, dtype=dtype)
    _, pos_indice, x_indice = np.intersect1d(pos, x, return_indices=True) 
    pos_val = torch.zeros((pos.shape[0], v.shape[1]), device=v.device)
    pos_val[pos_indice] = v[x_indice]
    return torch.cat([pos_val, xx], dim=1)

def check_numeric_issue(x):
    if torch.isnan(x).any():
        print("NAN detected")
        exit()
    if torch.isinf(x).any():
        print("INF detected")
        exit()
    if torch.abs(x).max() > 1e10:
        print("Large value detected")
        exit()

class GCN(nn.Module):
    
    def __init__(self, input, hidden, relu=True):
        super(GCN, self).__init__()
        self.conv = GCNConv(input, hidden)
        self.norm = nn.LayerNorm(hidden)
        #self.norm2 = GraphNorm(hidden)
        self.dp = nn.Dropout(args.dp, inplace=True)
        self.relu = nn.ReLU() if relu else nn.Identity()

    def forward(self, x, ei):
        adj = torch_sparse.SparseTensor(row=ei[0], col=ei[1], sparse_sizes=(x.size(0), x.size(0)))
        x = self.conv(x, adj)
        x = self.norm(x)
        #x = self.norm2(x)
        x = self.dp(x)
        return self.relu(x)

class LocalFWL(nn.Module):

    def __init__(self, input, hidden):
        super(LocalFWL, self).__init__()
        self.conv = GCNConv(input, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.mlp1 = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            #GraphNorm(hidden),
            nn.Dropout(args.dp, inplace=True),
            nn.ReLU(),
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            #GraphNorm(hidden),
            nn.Dropout(args.dp, inplace=True),
            nn.ReLU(),
        )
        self.mlp3 = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.Dropout(args.dp, inplace=True),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward_routine(self, x, ei, pos):
        x = self.conv(x, ei)
        x = self.conv2(x, ei)
        xx = x[pos[0]] * x[pos[1]]
        x = x[ei[0]] * x[ei[1]]
        x1 = self.mlp1(x)
        x = self.mlp2(x)
        #check_numeric_issue(x)
        return x, x1, xx

    def sparse_forward(self, x, ei, pos):
        n = x.shape[0]
        x, x1, xx = self.forward_routine(x, ei, pos)
        #print(f"memory2: {torch.cuda.memory_allocated(device = x.device) / 1024 / 1024 / 1024}", flush = True)
        x, v = sparse_bmm(ei, x, ei, x1, n)
        #print(f"memory3: {torch.cuda.memory_allocated(device = x.device) / 1024 / 1024 / 1024}", flush = True)
        x = combine(x, v, xx, pos)
        #check_numeric_issue(x)
        return self.mlp3(x).squeeze()
    
    def dense_forward(self, x, ei, pos):
        n = x.shape[0]
        x, x1, xx = self.forward_routine(x, ei, pos)
        #print(f"memory2: {torch.cuda.memory_allocated(device = x.device) / 1024 / 1024 / 1024}", flush = True)
        x, v = dense_bmm(ei, x, ei, x1, n)
        #print(f"memory3: {torch.cuda.memory_allocated(device = x.device) / 1024 / 1024 / 1024}", flush = True)
        x = combine(x, v, xx, pos)
        #check_numeric_issue(x)
        return self.mlp3(x).squeeze()

    def forward(self, x, ei, pos):
        #check_numeric_issue(x)
        #print("memory1.1:", torch.cuda.memory_allocated(device = x.device) / 1024 / 1024 / 1024, flush = True)
        pos_nodes = torch.cat([torch.unique(pos[0]), torch.unique(pos[1])], dim=0)
        subset, ei, new_pos_nodes, _ = k_hop_subgraph(pos_nodes, 1, ei, relabel_nodes=True)
        mapping = torch.full((pos.max().item() + 1, ), -1, dtype=torch.long, device=pos.device)
        mapping[pos_nodes] = new_pos_nodes
        pos = mapping[pos]
        x = x[subset]
        #print(subset.shape, ei.shape, flush = True)
        if ei.size(1) > args.edge_threhold:
            print("drop triggered", ei.size(1), flush = True)
            ei = ei[:, torch.randperm(ei.size(1))[:args.edge_threhold // 2]]
            ei = torch.cat([ei, pos], dim=1)
            ei = to_undirected(ei)
            #print(ei.shape, flush = True)
        ratio = ei.size(1) / subset.size(0) / (subset.size(0) - 1)
        #print(ratio, flush = True)
        #print("memory1:", torch.cuda.memory_allocated(device = x.device) / 1024 / 1024 / 1024, flush = True)
        if ratio <= args.dense_ratio:
            res = self.sparse_forward(x, ei, pos)
        else:
            res = self.dense_forward(x, ei, pos)
        #check_numeric_issue(res)
        return res

def train(model, optimizer, g, train_pos_edge, embedding = None):
    st = time.time()
    model.train()
    loader = DataLoader(range(train_pos_edge.size(1)), batch_size=args.batchsize, shuffle=True)
    total_loss = 0
    cnt = 0
    train_neg_edge = negative_sampling(g.edge_index, g.num_nodes)
    for mask in loader:
        cnt += 1
        if cnt % 100 == 0:
            '''val_res = eval(model, g, valid_pos_edge, valid_neg_edge)
            test_res = test(model, g, test_pos_edge, test_neg_edge)
            print(f"valid: hits@20: {val_res['hits@20']:.4f}, hits@50: {val_res['hits@50']:.4f}, hits@100: {val_res['hits@100']:.4f}")
            print(f"test: hits@20: {test_res['hits@20']:.4f}, hits@50: {test_res['hits@50']:.4f}, hits@100: {test_res['hits@100']:.4f}")
            model.train()'''
            print(f"train time: {time.time() - st:.4f}", flush = True)
            print(f"Batch {cnt}/{len(loader)}", flush=True)
        pos_edge = train_pos_edge[:, mask]
        neg_edge = train_neg_edge[:, mask]
        
        pred_pos = model(g.x, g.edge_index, pos_edge)
        pred_neg = model(g.x, g.edge_index, neg_edge)

        pos_loss = -F.logsigmoid(pred_pos).mean()
        neg_loss = -F.logsigmoid(-pred_neg).mean()
        loss = pos_loss + neg_loss
        optimizer.zero_grad()
        #print(f"memory4: {torch.cuda.memory_allocated(device = g.x.device) / 1024 / 1024 / 1024}", flush = True)
        loss.backward()
        if args.dataset == 'ogbl-ddi':
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            nn.utils.clip_grad_norm_(embedding.parameters(), 1.0)
        if args.dataset == 'ogbl.collab':
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    result = {'hits@20': 0, 'hits@50': 0, 'hits@100': 0}
    for k in 20, 50, 100:
        evaluator.K = k
        result[f'hits@{k}'] = evaluator.eval({
            'y_pred_pos': pred_pos,
            'y_pred_neg': pred_neg,
        })[f'hits@{k}']
    en = time.time()
    print(f"train time: {en - st:.4f}", flush = True)
    return total_loss / len(loader), result

def eval(model, g, val_pos_edge, val_neg_edge):
    st = time.time()
    model.eval()
    with torch.no_grad():
        loader = DataLoader(range(val_pos_edge.size(1)), batch_size=args.batchsize)
        pos_score = []
        cnt = 0
        for mask in loader:
            cnt += 1
            if cnt % 100 == 0:
                print(f"eval time: {time.time() - st:.4f}", flush = True)
                print(f"Batch {cnt}/{len(loader)}", flush=True)
            pos_score.append(model(g.x, g.edge_index, val_pos_edge[:, mask]))
        pos_score = torch.cat(pos_score, dim=0)
        loader = DataLoader(range(val_neg_edge.size(1)), batch_size=args.batchsize)
        neg_score = []
        cnt = 0
        for mask in loader:
            cnt += 1
            if cnt % 100 == 0:
                print(f"eval time: {time.time() - st:.4f}", flush = True)
                print(f"Batch {cnt}/{len(loader)}", flush=True)
            neg_score.append(model(g.x, g.edge_index, val_neg_edge[:, mask]))
        neg_score = torch.cat(neg_score, dim=0)
        result = {'hits@20': 0, 'hits@50': 0, 'hits@100': 0}
        if args.dataset == 'ogbl-citation2':
            neg_score = neg_score.view(-1, 1000)
            result['MRR'] = evaluator.eval({
                'y_pred_pos': pos_score,
                'y_pred_neg': neg_score,
            })['mrr_list'].mean().item()
        else:
            for k in 20, 50, 100:
                evaluator.K = k
                result[f'hits@{k}'] = evaluator.eval({
                    'y_pred_pos': pos_score,
                    'y_pred_neg': neg_score,
                })[f'hits@{k}']
        en = time.time()
        print(f"eval time: {en - st:.4f}", flush = True)
        return result

def test(model, g, test_pos_edge, test_neg_edge):
    st = time.time()
    model.eval()
    with torch.no_grad():
        loader = DataLoader(range(test_pos_edge.size(1)), batch_size=args.batchsize)
        pos_score = []
        cnt = 0
        for mask in loader:
            cnt += 1
            if cnt % 100 == 0:
                print(f"test time: {time.time() - st:.4f}", flush = True)
                print(f"Batch {cnt}/{len(loader)}", flush=True)
            pos_score.append(model(g.x, g.full_edge_index, test_pos_edge[:, mask]))
        pos_score = torch.cat(pos_score, dim=0)
        loader = DataLoader(range(test_neg_edge.size(1)), batch_size=args.batchsize)
        neg_score = []
        cnt = 0
        for mask in loader:
            cnt += 1
            if cnt % 100 == 0:
                print(f"test time: {time.time() - st:.4f}", flush = True)
                print(f"Batch {cnt}/{len(loader)}", flush = True)
            neg_score.append(model(g.x, g.full_edge_index, test_neg_edge[:, mask]))
        neg_score = torch.cat(neg_score, dim=0)
        result = {'hits@20': 0, 'hits@50': 0, 'hits@100': 0}
        if args.dataset == 'ogbl-citation2':
            neg_score = neg_score.view(-1, 1000)
            result['MRR'] = evaluator.eval({
                'y_pred_pos': neg_score,
                'y_pred_neg': pos_score,
            })['mrr_list'].mean().item()
        else:
            for k in 20, 50, 100:
                evaluator.K = k
                result[f'hits@{k}'] = evaluator.eval({
                    'y_pred_pos': pos_score,
                    'y_pred_neg': neg_score,
                })[f'hits@{k}']
        en = time.time()
        print(f"test time: {en - st:.4f}", flush = True)
        return result


st = time.time()
dataset = PygLinkPropPredDataset(name=args.dataset)
device = torch.device('cuda', args.gpu) if torch.cuda.is_available() else torch.device('cpu')

split_edge = dataset.get_edge_split()

if args.dataset =="ogbl-citation2":
    for name in ['train','valid','test']:
        u=split_edge[name]["source_node"]
        v=split_edge[name]["target_node"]
        split_edge[name]['edge']=torch.stack((u,v),dim=0).t()
    for name in ['valid','test']:
        u=split_edge[name]["source_node"].repeat(1, 1000).view(-1)
        v=split_edge[name]["target_node_neg"].view(-1)
        split_edge[name]['edge_neg']=torch.stack((u,v),dim=0).t()  

train_pos_edge = split_edge['train']['edge'].t().to(device)
test_pos_edge = split_edge['test']['edge'].t().to(device)
test_neg_edge = split_edge['test']['edge_neg'].t().to(device)
valid_pos_edge = split_edge['valid']['edge'].t().to(device)
valid_neg_edge = split_edge['valid']['edge_neg'].t().to(device)

g = dataset[0]
g.edge_index, _ = add_self_loops(g.edge_index, num_nodes=g.num_nodes)
g.edge_index = to_undirected(g.edge_index)
g.adj_t = g.edge_index
g = g.to(device)
if args.dataset == 'ogbl-ddi' or args.dataset == 'ogbl-ppa':
    embedding = torch.nn.Embedding(g.num_nodes, args.hidden).to(device)
    torch.nn.init.orthogonal_(embedding.weight)
    g.x = embedding.weight
else:
    embedding = None
    
model = LocalFWL(g.x.shape[1], args.hidden).to(device)
if args.dataset == 'ogbl-ddi':
    parameter = itertools.chain(model.parameters(), embedding.parameters())
else:
    parameter = model.parameters()
optimizer = torch.optim.Adam(parameter, lr=args.lr)
if args.step_lr_decay:
    lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=args.interval, gamma=args.gamma)

evaluator = Evaluator(name=args.dataset)

losses = []
valid_list = []
test_list = []
best_val = 0
best_epoch = 0
final_test_result = 0
en = time.time()
print(f"preprocess time: {en - st:.4f}", flush = True)
#print("memory0:", torch.cuda.memory_allocated(device = device) / 1024 / 1024 / 1024, flush = True)
for epoch in range(args.epochs):
    if args.dataset == 'ogbl-collab':
        u, v = valid_pos_edge
        re_index = torch.stack([v, u], dim=0)
        g.full_edge_index = torch.cat([g.edge_index, re_index], dim=1)
    else:
        g.full_edge_index = g.edge_index
    loss, result = train(model, optimizer, g, train_pos_edge, embedding)
    losses.append(loss)
    if args.step_lr_decay:
        lr_scheduler.step()
    val_res = eval(model, g, valid_pos_edge, valid_neg_edge)
    valid_list.append(val_res[args.metric])
    test_res = test(model, g, test_pos_edge, test_neg_edge)
    test_list.append(test_res[args.metric])
    if val_res[args.metric] > best_val:
        best_val = val_res[args.metric]
        best_epoch = epoch
        final_test_result = test_res[args.metric]
    if epoch - best_epoch >= 50:
        break
    print(f"Epoch {epoch}, Loss: {loss:.4f}, Valid hit: {val_res[args.metric]:.4f}, Test hit: {test_res[args.metric]:.4f}", flush = True)
    print(f"train: hits@20: {result['hits@20']:.4f}, hits@50: {result['hits@50']:.4f}, hits@100: {result['hits@100']:.4f}", flush = True)
    print(f"valid: hits@20: {val_res['hits@20']:.4f}, hits@50: {val_res['hits@50']:.4f}, hits@100: {val_res['hits@100']:.4f}", flush = True)
    print(f"test: hits@20: {test_res['hits@20']:.4f}, hits@50: {test_res['hits@50']:.4f}, hits@100: {test_res['hits@100']:.4f}", flush = True)

print(f"Test hit: {final_test_result:.4f}")

plt.figure()
plt.plot(range(len(losses)), losses, label='loss')
plt.plot(range(len(losses)), valid_list, label='valid')
plt.plot(range(len(losses)), test_list, label='test')
plt.xlabel('epoch')
plt.ylabel('metric')
plt.legend()
plt.savefig("result/plot-" + args.dataset + str(args.gpu) + ".png")
