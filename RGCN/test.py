import os
import math
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch_scatter import scatter_add
from torch_geometric.data import Data

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm.std import trange

#load data
# use_cuda = args.gpu >= 0 and torch.cuda.is_available()
# if use_cuda:
#     torch.cuda.set_device(args.gpu)

from utils import generate_sampled_graph_and_labels

from model import RGCN


def train(train_triplets, model, batch_size, split_size, negative_sample, reg_ratio, num_entities, num_relations):

    train_data = generate_sampled_graph_and_labels(train_triplets, batch_size, split_size, num_entities, num_relations, negative_sample)

    entity_embedding = model(train_data.entity, train_data.edge_index, train_data.edge_type, train_data.edge_norm)
    loss = model.score_loss(entity_embedding, train_data.samples, train_data.labels) + reg_ratio * model.reg_loss(entity_embedding)

    return loss
best_mrr = 0

with open(os.path.join('./data', 'entities.dict')) as f:
    entity2id = dict()

    for line in f:
        eid, entity = line.strip().split('\t')
        entity2id[entity] = int(eid)

with open(os.path.join('./data', 'relations.dict')) as f:
        relation2id = dict()

        for line in f:
            rid, relation = line.strip().split('\t')
            relation2id[relation] = int(rid)

#read_triplets
train_triplets = []
with open('data/train.txt') as f:
    for line in f:
        head, relation, tail = line.strip().split('\t')
        train_triplets.append((entity2id[head], relation2id[relation], entity2id[tail]))
train_triplets = np.array(train_triplets)
valid_triplets = []
with open('data/valid.txt') as f:
    for line in f:
        head, relation, tail = line.strip().split('\t')
        valid_triplets.append((entity2id[head], relation2id[relation], entity2id[tail]))
valid_triplets = np.array(valid_triplets)

test_triplets = []

with open('data/test.txt') as f:
    for line in f:
        head, relation, tail = line.strip().split('\t')
        test_triplets.append((entity2id[head], relation2id[relation], entity2id[tail]))
test_triplets = np.array(test_triplets)
all_triplets = torch.LongTensor(np.concatenate((train_triplets,valid_triplets,test_triplets)))



#build graph
def build_test_graph(num_nodes, num_rels, triplets):
    src, rel, dst = triplets.transpose()

    src = torch.from_numpy(src)
    rel = torch.from_numpy(rel)
    dst = torch.from_numpy(dst)

    src, dst = torch.cat((src, dst)), torch.cat((dst, src))
    rel = torch.cat((rel, rel + num_rels))

    edge_index = torch.stack((src, dst))
    edge_type = rel

    data = Data(edge_index = edge_index.to(torch.long))
    data.entity = torch.from_numpy(np.arange(num_nodes))
    data.edge_type = edge_type
    data.edge_norm = edge_normalization(edge_type, edge_index, num_nodes, num_rels)

    return data

def edge_normalization(edge_type, edge_index, num_entity, num_relation):
    '''
        Edge normalization trick
        - one_hot: (num_edge, num_relation)
        - deg: (num_node, num_relation)
        - index: (num_edge)
        - deg[edge_index[0]]: (num_edge, num_relation)
        - edge_norm: (num_edge)
    '''
    one_hot = F.one_hot(edge_type.to(torch.int64), num_classes = 2 * num_relation).to(torch.float)
    deg = scatter_add(one_hot, edge_index[0].to(torch.int64), dim = 0, dim_size = num_entity)
    index = edge_type + torch.arange(len(edge_index[0])) * (2 * num_relation)
    edge_norm = 1 / deg[edge_index.long()[0]].view(-1)[index.long()]

    return edge_norm

test_graph = build_test_graph(len(entity2id), len(relation2id), train_triplets)
print(test_graph)
valid_triplets = torch.LongTensor(valid_triplets)
test_triplets = torch.LongTensor(test_triplets)


#model 
model = RGCN(len(entity2id), len(relation2id), num_bases = 4, dropout = 0.2)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)
history = pd.DataFrame(columns=['train_loss', 'epoch'])
for epoch in trange(1, 1000, desc = 'Epochs', position = 0):
    model.train()
    optimizer.zero_grad()

    loss = train(train_triplets, model, batch_size = 400, split_size = 0.5, negative_sample=1,reg_ratio=1e-2, num_entities=len(entity2id), num_relations=len(relation2id))

    tqdm.write("Train Loss {} at epoch {}".format(loss, epoch))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    #validate
    #test  
# calc_mrr(entity_embedding, model.relation_embedding, train_triplets, test_triplets, valid_triplets, hits=[1, 3, 10])
# def calc_mrr(embedding, w, train_triplets, valid_triplets, test_triplets, hits=[], eval_bz=100, eval_p="filtered"):
# def get_score_all(embedding, w, train_triplets, other_triplets, test_triplets, entity2id):
# test(test_triplets, model,train_triplets, test_graph, valid_triplets)
torch.save(model,os.path.join('output/' ,'modelnew.pt'))
#torch.save(to_save, os.path.join(output_dir, model_file_save))
model = torch.load('output/modelnew.pt')

#prediction function
entity_embedding = model(test_graph.entity, test_graph.edge_index, test_graph.edge_type, test_graph. edge_norm)
from utils import get_score_all
list_123 = get_score_all(entity_embedding, model.relation_embedding, train_triplets, valid_triplets, test_triplets, entity2id)
print(list_123)


