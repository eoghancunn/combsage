import torch
from sklearn import metrics
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as MF
import dgl
import dgl.nn as dglnn
import pandas as pd
import networkx as nx
from dgl.dataloading.base import EdgePredictionSampler
from dgl.dataloading.base import Mapping, NID, EID, heterograph, compact_graphs, LazyFeature, find_exclude_eids
from dgl.dataloading import NeighborSampler
from dgl.transforms.functional import to_block

class HetEdgePredictionSampler(EdgePredictionSampler):
    
    """Class to override the negative sampling in the EdgePredictionSampler
    Takes an additional argument: g_homo -- a homogenous version of the 
    heterogenous graph, so that negative edges are sampled unifromly across edge types
    """
    
    def __init__(self, sampler, g_homo = None, exclude = None, reverse_eids = None, 
                reverse_etypes = None, negative_sampler = None, prefetch_labels = None):
        EdgePredictionSampler.__init__(self, sampler, exclude, reverse_eids, 
                reverse_etypes, negative_sampler, prefetch_labels)
        self.g_homo = g_homo
        
    def sample(self, g, seed_edges):    # pylint: disable=arguments-differ
        """Samples a list of blocks, as well as a subgraph containing the sampled
        edges from the original graph.

        If :attr:`negative_sampler` is given, also returns another graph containing the
        negative pairs as edges.
        """
        if isinstance(seed_edges, Mapping):
            seed_edges = {g.to_canonical_etype(k): v for k, v in seed_edges.items()}
        exclude = self.exclude
        pair_graph = g.edge_subgraph(
            seed_edges, relabel_nodes=False, output_device=self.output_device)
        eids = pair_graph.edata[EID]
        
        homo_edges = []
        offset = 0
        for k in seed_edges:
            homo_edges.append(seed_edges[k] + offset)
            offset += g.number_of_edges(etype = k)
        homo_edges = torch.cat(homo_edges)
        
        if self.negative_sampler is not None:
            neg_graph = self._build_neg_graph(self.g_homo, homo_edges)
            pair_graph, neg_graph = compact_graphs([pair_graph, neg_graph])
        else:
            pair_graph = compact_graphs(pair_graph)

        pair_graph.edata[EID] = eids
        seed_nodes = pair_graph.ndata[NID]

        exclude_eids = find_exclude_eids(
            g, seed_edges, exclude, self.reverse_eids, self.reverse_etypes,
            self.output_device)

        input_nodes, _, blocks = self.sampler.sample(g, seed_nodes, exclude_eids)

        if self.negative_sampler is None:
            return self.assign_lazy_features((input_nodes, pair_graph, blocks))
        else:
            return self.assign_lazy_features((input_nodes, pair_graph, neg_graph, blocks))
        
        
class HomoNeighborSampler(NeighborSampler):
    
    def sample_blocks(self, g, seed_nodes, exclude_eids=None):
        output_nodes = seed_nodes
        blocks = []
        # convert to homogeneous graph first
        # NOTE pass the list of feature names in data fields
        homo_g = dgl.to_homogeneous(g, ndata=['feat'])
        for fanout in reversed(self.fanouts):
            # sample on the homogeneous graph
            frontier = homo_g.sample_neighbors(
                seed_nodes, fanout, edge_dir=self.edge_dir, prob=self.prob,
                replace=self.replace, output_device=self.output_device,
                exclude_edges=exclude_eids)
            # revert the subgraph to heterogeneous graph before creating the block
            frontier = dgl.to_heterogeneous(frontier, g.ntypes, g.etypes)
            eid = frontier.edata[EID]
            block = to_block(frontier, seed_nodes)
            block.edata[EID] = eid
            seed_nodes = block.srcdata[NID]
            blocks.insert(0, block)

        return seed_nodes, output_nodes, blocks


def convert_to_heterograph(G, n_nodes = False):
    edges = []
    if not n_nodes:
        num_nodes_dict = {'paper': G.number_of_nodes()}
    else: num_nodes_dict = {'paper': n_nodes}

    for i,f in enumerate(G.nodes()):
        G_ego = nx.ego_graph(G,f)
        G_ego_less_f = nx.subgraph(G,G.neighbors(f))
        groups = nx.connected_components(G_ego_less_f)
        for i,group in enumerate(groups): 
            for n in group: 
                edge = {'src':n, 'type':str(i+1), 'dst':f}
                edges.append(edge)
    
    het_edge_df = pd.DataFrame(edges)
    edge_dict = {}
    for edge_type in het_edge_df['type'].unique():
        edges = het_edge_df[het_edge_df['type'] == edge_type]
        src = torch.from_numpy(edges['src'].values)
        dst = torch.from_numpy(edges['dst'].values)
        edge_dict[('paper',str(edge_type),'paper')] = (src,dst)
    
    return dgl.heterograph(edge_dict, num_nodes_dict=num_nodes_dict)


def convert_to_heterograph_group_isolates(G, n_nodes = False):
    edges = []
    if not n_nodes:
        num_nodes_dict = {'paper': G.number_of_nodes()}
    else: num_nodes_dict = {'paper': n_nodes}

    for i,f in enumerate(G.nodes()):
        G_ego = nx.ego_graph(G,f)
        G_ego_less_f = nx.subgraph(G,G.neighbors(f))
        groups = nx.connected_components(G_ego_less_f)
        group_index = 2
        for i,group in enumerate(groups): 
            if len(group) == 1:
                label = 1
            else:
                label = group_index
                group_index += 1
            for n in group: 
                edge = {'src':n, 'type':str(label), 'dst':f}
                edges.append(edge)
    
    het_edge_df = pd.DataFrame(edges)
    edge_dict = {}
    for edge_type in het_edge_df['type'].unique():
        edges = het_edge_df[het_edge_df['type'] == edge_type]
        src = torch.from_numpy(edges['src'].values)
        dst = torch.from_numpy(edges['dst'].values)
        edge_dict[('paper',str(edge_type),'paper')] = (src,dst)
    
    return dgl.heterograph(edge_dict, num_nodes_dict=num_nodes_dict)



def evaluate(model, graph, val_pos_g, val_neg_g, train_pos_g = None, train_neg_g = None, device = 'cpu'):

    def compute_loss(pos_score, neg_score): 
        scores = torch.cat([pos_score, neg_score])
        labels = torch.cat([torch.ones_like(pos_score),
                            torch.zeros_like(neg_score)])
        loss = F.binary_cross_entropy_with_logits(scores, labels)
        return loss

    def compute_auc(pos_score, neg_score): 
        scores = torch.cat([pos_score, neg_score]).numpy()
        labels = torch.cat([torch.ones(pos_score.shape[0]),
                            torch.zeros(neg_score.shape[0])]).numpy()
        return metrics.roc_auc_score(labels, scores)
    
    with torch.no_grad():
        emb = model.inference(graph,device,4096,0,'cpu')
        
        pos_src, pos_dst = val_pos_g.edges()
        neg_src, neg_dst = val_neg_g.edges()
        
        pos_score = model.predict(emb[pos_src], emb[pos_dst])
        neg_score = model.predict(emb[neg_src], emb[neg_dst])
        
        val_loss = compute_loss(pos_score,neg_score)
        val_auc = compute_auc(pos_score,neg_score)
        
        if train_pos_g and train_neg_g: 
            
            pos_src, pos_dst = train_pos_g.edges()
            neg_src, neg_dst = train_neg_g.edges()
            train_pos_score = model.predict(emb[pos_src],
                                            emb[pos_dst])
            train_neg_score = model.predict(emb[neg_src],
                                            emb[neg_dst])
            train_loss = compute_loss(train_pos_score,
                                      train_neg_score)
            train_auc = compute_auc(train_pos_score,train_neg_score)
        else: 
            train_loss, train_auc = None, None
        
    return val_loss, val_auc, train_loss, train_auc

def preprocess_graph(edge_list, X):
    
    #convert to undir g
    edge_list.columns = ['source', 'target']
    g_nx = nx.from_pandas_edgelist(edge_list)
    edge_list = nx.to_pandas_edgelist(g_nx)
    
    src = edge_list['source'].to_numpy()
    dst = edge_list['target'].to_numpy()
    
    g = dgl.graph((src,dst))
    g.ndata['feat'] = torch.tensor(X).float()
    
    