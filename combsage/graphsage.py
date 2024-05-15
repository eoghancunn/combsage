import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as MF
import dgl
import dgl.nn as dglnn
import tqdm

class SAGE(nn.Module):
    def __init__(self, in_feats, n_hidden, agg, dropout = 0):
        super().__init__()
        self.n_hidden = n_hidden
        self.dropout = dropout
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, agg, feat_drop = self.dropout))
        self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, agg, feat_drop = self.dropout))
        # self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, 'mean'))
        self.predictor = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, 1))

    def predict(self, h_src, h_dst):
        return self.predictor(h_src * h_dst)

    def forward(self, pair_graph, neg_pair_graph, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = F.relu(h)
        pos_src, pos_dst = pair_graph.edges()
        neg_src, neg_dst = neg_pair_graph.edges()
        h_pos = self.predict(h[pos_src], h[pos_dst])
        h_neg = self.predict(h[neg_src], h[neg_dst])
        return h_pos, h_neg

    def inference(self, g, device, batch_size, num_workers, buffer_device=None):
        # The difference between this inference function and the one in the official
        # example is that the intermediate results can also benefit from prefetching.
        feat = g.ndata['feat']
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1, prefetch_node_feats=['feat'])
        dataloader = dgl.dataloading.DataLoader(
                g, torch.arange(g.num_nodes()).to(g.device), sampler, device=device,
                batch_size=1000, shuffle=False, drop_last=False, num_workers=num_workers)
        if buffer_device is None:
            buffer_device = device

        for l, layer in enumerate(self.layers):
            y = torch.zeros(g.num_nodes(), self.n_hidden, device=buffer_device)
            feat = feat.to(device)

            for input_nodes, output_nodes, blocks in dataloader:
                x = feat[input_nodes]
                h = layer(blocks[0], x)
                if l != len(self.layers) - 1:
                    h = F.relu(h)
                y[output_nodes] = h.to(buffer_device)
            feat = y
        return y
