import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as MF
import dgl
import dgl.nn as dglnn
import tqdm

class IDRSAGE(nn.Module):
    
    def __init__(self, in_feats, n_hidden, n_types, comm_agg = 'lstm', dropout = 0.1,
                 isolates_agg = 'pool', inter_agg = 'max', k = 1):
        super().__init__()
        self.n_hidden = n_hidden
        self.n_types = n_types
        
        self.comm_agg = dglnn.SAGEConv(in_feats, n_hidden, comm_agg,
                                       feat_drop = dropout)
        self.isolates_agg = dglnn.SAGEConv(in_feats, n_hidden, isolates_agg,
                                           feat_drop = dropout)
        
        if inter_agg == 'lstm':
            self.lstm1 = nn.LSTM(n_hidden, n_hidden, batch_first = True)
            self.agg_func1 = self._lstm_reducer1
        else: 
            self.agg_func1 = self._max_reducer
            
        if k == 2: 
            self.comm_agg_2 = dglnn.SAGEConv(n_hidden, n_hidden, comm_agg,
                                             feat_drop = 0.2)
            self.isolates_agg_2 = dglnn.SAGEConv(n_hidden, n_hidden, isolates_agg,
                                                 feat_drop = 0.2)

            if inter_agg == 'lstm':
                self.lstm2 = nn.LSTM(n_hidden, n_hidden, batch_first = True)
                self.agg_func2 = self._lstm_reducer2
            else: 
                self.agg_func2 = self._max_reducer
        
        self.layers = nn.ModuleList()
        
        module_dict_k_1 = {str(i): self.comm_agg for i in range(2,n_types+1)}
        module_dict_k_1.update({'1': self.isolates_agg})
        
        self.layers.append(dglnn.HeteroGraphConv(
            module_dict_k_1, aggregate = self.agg_func1))
        
        if k == 2:
            module_dict_k_2 = {str(i): self.comm_agg_2 for i in range(2,n_types+1)}
            module_dict_k_2.update({'1': self.isolates_agg_2})

            self.layers.append(dglnn.HeteroGraphConv(
                module_dict_k_2, aggregate = self.agg_func2))
            
        self.predictor = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden), 
            nn.ReLU(),
            nn.Linear(n_hidden, 1)) 
        
    def _lstm_reducer1(self, tensors, dsttype):
        
        # with torch.no_grad():
        masks = [torch.cat((torch.ones_like(torch.unique(t,dim = 0)[:-1]),
                              torch.zeros_like(t)))[:t.shape[0]]
                    for t in tensors]
        stack = torch.stack([t*m for m,t in zip(masks,tensors)],dim=1)
        _,(rst,_) = self.lstm1(stack)
        return rst.squeeze(0)
        
    def _lstm_reducer2(self, tensors, dsttype):
        
        # with torch.no_grad():
        masks = [torch.cat((torch.ones_like(torch.unique(t,dim = 0)[:-1]),
                              torch.zeros_like(t)))[:t.shape[0]]
                    for t in tensors]
        stack = torch.stack([t*m for m,t in zip(masks,tensors)],dim=1)
        _,(rst,_) = self.lstm2(stack)
        return rst.squeeze(0)  
        
    def _max_reducer(self, tensors, dsttype):
        
        with torch.no_grad():
            masks = [torch.cat((torch.ones_like(torch.unique(t,dim = 0)[:-1]),
                                  torch.zeros_like(t)))[:t.shape[0]]
                        for t in tensors]
        stack = torch.stack([t*m for m,t in zip(masks,tensors)],dim=1)
        return torch.max(stack, dim = 1).values
        
    def predict(self, h_src, h_dst): 
        return self.predictor(h_src*h_dst)       

    def forward(self, pos_graph, neg_graph, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers,blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1: 
                h = {k:F.relu(h[k]) for k in h}
        h = h['paper']

        pos_e_tensors = [pos_graph.edges(etype = etype)
                     for etype in pos_graph.canonical_etypes]
        neg_e_tensors = [neg_graph.edges(etype = etype)
                     for etype in neg_graph.canonical_etypes]
        
        
        pos_src = torch.hstack([e[0] for e in pos_e_tensors])
        pos_dst = torch.hstack([e[1] for e in pos_e_tensors])
        
        neg_src = torch.hstack([e[0] for e in neg_e_tensors])
        neg_dst = torch.hstack([e[1] for e in neg_e_tensors])
        
        h_pos = self.predict(h[pos_src], h[pos_dst])
        h_neg = self.predict(h[neg_src], h[neg_dst])

        return h_pos, h_neg
            
    def inference(self, g, device, batch_size, num_workers, buffer_device=None):
        # The difference between this inference function and the one in the official
        # example is that the intermediate results can also benefit from prefetching.
        feat = g.ndata['feat']
        edge_dict = {etype: g.edges(etype = etype, form = 'all')[-1] for etype in g.etypes}
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1, prefetch_node_feats=['feat'])
        dataloader = dgl.dataloading.DataLoader(
                g, torch.arange(g.num_nodes()).to(g.device), sampler, device=device,
                batch_size=1000, shuffle=False, drop_last=False, num_workers=num_workers)
        if buffer_device is None:
            buffer_device = device

        for l, layer in enumerate(self.layers):
            y = torch.zeros(g.num_nodes(), self.n_hidden, device=buffer_device)
            feat = feat.to(device)

            with tqdm.tqdm(dataloader) as tq:
                for input_nodes, output_nodes, blocks in tq:
                    tq.set_description('Inference.')
                    x = {'paper':feat[input_nodes]}
                    h = layer(blocks[0], x)['paper']
                    if l != len(self.layers) -1: 
                        h = F.relu(h)
                    y[output_nodes] = h.to(buffer_device)
                feat = y
        
        return y

    
class IDRSAGEJK(IDRSAGE):
    
    def __init__(self, in_feats, n_hidden, n_types, comm_agg = 'pool',
                 isolates_agg = 'pool', inter_agg = 'lstm', jk_mode = 'cat', dropout = 0.1):
        super().__init__(in_feats = in_feats, n_hidden = n_hidden, n_types = n_types, 
                        comm_agg = comm_agg, isolates_agg = isolates_agg, inter_agg = inter_agg, k = 2, dropout = dropout)
        self.jumping_knowledge = dglnn.pytorch.utils.JumpingKnowledge(mode = jk_mode)
        self.predictor = nn.Sequential(
            nn.Linear(2*n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden), 
            nn.ReLU(),
            nn.Linear(n_hidden, 1)) 
        
    def forward(self, pos_graph, neg_graph, blocks, x):
        
        h = x
        h1 = self.layers[0](blocks[0],h)
        h2 = self.layers[1](blocks[1],h1)
        
        n = h2['paper'].shape[0]
        h1 = h1['paper'][:n,:]
        h2 = h2['paper']
        h_final = self.jumping_knowledge([h1,h2])
        
        pos_e_tensors = [pos_graph.edges(etype = etype)
                     for etype in pos_graph.canonical_etypes]
        neg_e_tensors = [neg_graph.edges(etype = etype)
                     for etype in neg_graph.canonical_etypes]
        
        
        pos_src = torch.hstack([e[0] for e in pos_e_tensors])
        pos_dst = torch.hstack([e[1] for e in pos_e_tensors])
        
        neg_src = torch.hstack([e[0] for e in neg_e_tensors])
        neg_dst = torch.hstack([e[1] for e in neg_e_tensors])
        
        h_pos = self.predict(h_final[pos_src], h_final[pos_dst])
        h_neg = self.predict(h_final[neg_src], h_final[neg_dst])

        return h_pos, h_neg
        
    def inference(self, g, device, batch_size, num_workers, buffer_device=None):
        # The difference between this inference function and the one in the official
        # example is that the intermediate results can also benefit from prefetching.
        feat = g.ndata['feat']
        edge_dict = {etype: g.edges(etype = etype, form = 'all')[-1] for etype in g.etypes}
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1, prefetch_node_feats=['feat'])
        dataloader = dgl.dataloading.DataLoader(
                g, torch.arange(g.num_nodes()).to(g.device), sampler, device=device,
                batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers)
        if buffer_device is None:
            buffer_device = device
        
        reprs = []
        
        for l, layer in enumerate(self.layers):
            y = torch.zeros(g.num_nodes(), self.n_hidden, device=buffer_device)
            feat = feat.to(device)
            with tqdm.tqdm(dataloader) as tq:
                for input_nodes, output_nodes, blocks in tq:
                    tq.set_description('Inference.')
                    x = {'paper':feat[input_nodes]}
                    h = layer(blocks[0], x)['paper']
                    y[output_nodes] = h.to(buffer_device)
                feat = y
            reprs.append(y)
        
        return self.jumping_knowledge(reprs)

    