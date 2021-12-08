import torch
import torch.nn as nn
from torch.nn import Parameter as Param, init


class GatedGraphConv(nn.Module):
    r"""The gated graph convolution operator from the `"Gated Graph Sequence
    Neural Networks" <https://arxiv.org/abs/1511.05493>`_ paper
    .. math::
        \mathbf{h}_i^{(0)} &= \mathbf{x}_i \, \Vert \, \mathbf{0}
        \mathbf{m}_i^{(l+1)} &= \sum_{j \in \mathcal{N}(i)} \mathbf{\Theta}
        \cdot \mathbf{h}_j^{(l)}
        \mathbf{h}_i^{(l+1)} &= \textrm{GRU} (\mathbf{m}_i^{(l+1)},
        \mathbf{h}_i^{(l)})
    up to representation :math:`\mathbf{h}_i^{(L)}`.
    The number of input channels of :math:`\mathbf{x}_i` needs to be less or
    equal than :obj:`out_channels`.
    Args:
        out_channels (int): Size of each input sample.
        num_layers (int): The sequence length :math:`L`.
        aggr (string): The aggregation scheme to use
            (:obj:`"add"`, :obj:`"mean"`, :obj:`"max"`).
            (default: :obj:`"add"`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
    """

    def __init__(self, input_dim, num_timesteps, num_edge_types, bias=True, dropout=0):
        super(GatedGraphConv, self).__init__()

        self._input_dim = input_dim
        self.num_timesteps = num_timesteps
        self.num_edge_types = num_edge_types

        self.weight = Param(torch.Tensor(num_timesteps, num_edge_types, input_dim, input_dim))
        self.bias = Param(torch.Tensor(num_timesteps, num_edge_types, input_dim))
        self.rnn = torch.nn.GRUCell(input_dim, input_dim, bias=bias)
        self.dropout = torch.nn.Dropout(dropout)

        self.reset_parameters()

    def reset_parameters(self):
        for t in range(self.num_timesteps):
            for e in range(self.num_edge_types):
                torch.nn.init.xavier_uniform_(self.weight[t, e])
        init.uniform_(self.bias, -0.01, 0.01)
        self.rnn.reset_parameters()

    def forward(self, x, adj_sparse_list):
        """"""
        if len(adj_sparse_list) != self.num_edge_types:
            raise ValueError(f'GatedGraphConv constructed with {self.num_edge_types} edge types, '
                             f'but {len(adj_sparse_list)} were passed')
        h = x
        if h.size(1) > self._input_dim:
            raise ValueError('The number of input channels is not allowed to '
                             'be larger than the number of output channels')

        if h.size(1) < self._input_dim:
            zero = h.new_zeros(h.size(0), self._input_dim - h.size(1))
            h = torch.cat([h, zero], dim=1)

        for t in range(self.num_timesteps):
            new_h = []
            for e in range(self.num_edge_types):
                if torch.sparse.sum(adj_sparse_list[e]) == 0:
                    continue
                m = self.dropout(torch.matmul(h, self.weight[t, e]) + self.bias[t, e])
                new_h.append(torch.sparse.mm(adj_sparse_list[e], m))
            m_sum = torch.sum(torch.stack(new_h), dim=0)
            h = self.rnn(m_sum, h)

        return h