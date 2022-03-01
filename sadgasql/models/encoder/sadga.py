import itertools
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sadgasql.models.encoder.gated_graph_conv import GatedGraphConv


# Adapted from The Annotated Transformer
def relative_attention_logits(query, key, relation):
    qk_matmul = torch.matmul(query, key.transpose(-2, -1))
    q_t = query.permute(0, 2, 1, 3)
    r_t = relation.transpose(-2, -1)
    q_tr_t_matmul = torch.matmul(q_t, r_t)
    q_tr_tmatmul_t = q_tr_t_matmul.permute(0, 2, 1, 3)
    return (qk_matmul + q_tr_tmatmul_t) / math.sqrt(query.shape[-1])


# Adapted from The Annotated Transformer
def relative_attention_values(weight, value, relation):
    wv_matmul = torch.matmul(weight, value)
    w_t = weight.permute(0, 2, 1, 3)
    w_tr_matmul = torch.matmul(w_t, relation)
    w_tr_matmul_t = w_tr_matmul.permute(0, 2, 1, 3)
    return wv_matmul + w_tr_matmul_t

# Adapted from The Annotated Transformer
def attention_with_relations(query, key, value, relation_k, relation_v, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = relative_attention_logits(query, key, relation_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn_orig = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn_orig)
    return relative_attention_values(p_attn, value, relation_v), p_attn_orig


def clones(module_fn, N):
    return nn.ModuleList([module_fn() for _ in range(N)])


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, layer_size, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer_size)

    def forward(self, x, relation, q_len, desc):
        "Pass the input (and mask) through each layer in turn."
        for index, layer in enumerate(self.layers):
            x = layer(x, relation, q_len, desc)
        return self.norm(x)


# Adapted from The Annotated Transformer
class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, sadga, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.sadga = sadga
        self.feed_forward = feed_forward
        self.sublayer = clones(lambda: SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, relation, q_len, desc):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.sadga(x, relation, q_len, desc))
        return self.sublayer[1](x, self.feed_forward)


# Adapted from The Annotated Transformer
class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class SadgaLayer(nn.Module):
    def __init__(self, hidden_size, ggnn_num_timesteps, ggnn_num_edge_types, num_relation_kinds, device, activation_func='tanh', dropout=0):
        super(SadgaLayer, self).__init__()
        self._device = device
        self._hidden_size = hidden_size
        self._ggnn_num_timesteps = ggnn_num_timesteps
        self._ggnn_num_edge_types = ggnn_num_edge_types

        self._ggnn = GatedGraphConv(hidden_size, ggnn_num_timesteps, ggnn_num_edge_types, dropout=dropout)
        self._struc_awr_aggr = StructureAwareGraphAggr(hidden_size, dropout, activation_func)

        self.rel_k_emb = nn.Embedding(num_relation_kinds, self._hidden_size)
        self.rel_v_emb = nn.Embedding(num_relation_kinds, self._hidden_size)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, input, relation, q_len, desc):
        num_nl = q_len
        num_db = input.size(1) - q_len

        # question-graph node input
        nl_input = input[:, :q_len]
        # schema-graph node input
        db_input = input[:, q_len:]

        # A. Dual-Graph Construction
        # question-graph construction relations
        nl_relation = relation[:num_nl, :num_nl]
        # schema-graph construction relations
        db_relation = relation[-num_db:, -num_db:]

        # B. Dual-Graph Encoding
        # relation node
        x_spadj_list = [self._add_relation_node(x, relation)
                        for x, relation in zip([nl_input, db_input], [nl_relation, db_relation])]
        rggnn_output = [self._ggnn(x, spadj) for x, spadj in x_spadj_list]
        nl_rggnn_output, db_rggnn_output = [rggnn_output[i][:num] for i, num in enumerate([num_nl, num_db])]

        # cross-graph relations
        nl_db_relation = relation[:q_len, q_len:]
        nl_db_relation_t = torch.LongTensor(nl_db_relation).to(self._device)
        nl_db_re_k = self.rel_k_emb(nl_db_relation_t)
        nl_db_re_v = self.rel_v_emb(nl_db_relation_t)

        # graph adj
        db_adj, nl_adj = np.eye(num_db), np.eye(num_nl)
        db_adj[(db_relation > 0)] = 1
        nl_adj[(nl_relation > 0)] = 1

        # add dependency tree linking
        for tuple in desc['word_adj_tuple']:
            nl_adj[tuple[0], tuple[1]] = 1

        # C. Structure-Aware Aggregation
        nl_output = self._struc_awr_aggr(nl_rggnn_output,
                                         db_rggnn_output,
                                         nl_db_re_k,
                                         nl_db_re_v,
                                         torch.LongTensor(db_adj).to(self._device))
        db_output = self._struc_awr_aggr(db_rggnn_output,
                                         nl_rggnn_output,
                                         torch.transpose(nl_db_re_k, 1, 0),
                                         torch.transpose(nl_db_re_v, 1, 0),
                                         torch.LongTensor(nl_adj).to(self._device))

        h = torch.cat([nl_output, db_output])

        return h

    def _add_relation_node(self, x, relation):
        num_item = x.size(1)
        relation_index = num_item

        adj_forward = []
        adj_back = []
        adj_self = []

        relation_ids = []
        for i, j in itertools.product(range(num_item), repeat=2):
            if i <= j:
                continue
            relation_id = relation[i][j]
            if relation_id != 0:
                relation_ids.append(relation_id)
                adj_forward.append((i, relation_index))
                adj_forward.append((relation_index, j))
                adj_back.append((j, relation_index))
                adj_back.append((relation_index, i))
                relation_index = relation_index + 1
        num_node = relation_index
        for i in range(num_node):
            adj_self.append((i, i))

        relations_t = torch.LongTensor(relation_ids).to(self._device)
        relation_emb = self.rel_v_emb(relations_t)

        input = torch.cat([x.squeeze(0), relation_emb], dim=0)
        all_adj_types = [adj_forward, adj_back, adj_self]

        adj_list = []
        for i in range(len(all_adj_types)):
            adj_t = input.new_zeros((input.size(0), input.size(0)))
            adj_tuple = all_adj_types[i]
            for edge in adj_tuple:
                adj_t[edge[1], edge[0]] = 1
            adj_list.append(adj_t.to_sparse())

        return input, adj_list


class StructureAwareGraphAggr(nn.Module):
    def __init__(self,
                 d_model,
                 dropout=0.5,
                 activation_func='tanh'):
        super(StructureAwareGraphAggr, self).__init__()

        self.d_model = d_model
        self.glob_linear = nn.Linear(d_model, d_model)
        self.glob_v_linears = clones(lambda: nn.Linear(d_model, d_model), 2)

        self.glob_query_linear = nn.Linear(d_model, d_model)
        self.loc_query_linear = nn.Linear(d_model, d_model)

        self.loc_gate_linear = nn.Linear(d_model * 2, 1)
        self.gate_linear = nn.Linear(d_model * 2, 1)
        self.dropout = nn.Dropout(p=dropout)
        self.activation_func = torch.tanh if activation_func == 'tanh' else torch.relu

    def forward(self,
                q_enc: torch.Tensor,
                k_enc: torch.Tensor,
                rel_k: torch.Tensor,
                rel_v: torch.Tensor,
                k_adj: torch.Tensor, ) -> torch.Tensor:
        q_num, k_num = q_enc.size(0), k_enc.size(0)

        # Global Pooling
        q_glob_enc = torch.mean(q_enc, dim=0)
        k_s = torch.matmul(self.glob_linear(q_glob_enc), torch.transpose(k_enc, 1, 0)) / math.sqrt(self.d_model)
        k_s = torch.sigmoid(k_s).unsqueeze(1)
        g_1, k_1 = [l(x) for l, x in zip(self.glob_v_linears, (q_glob_enc, k_enc))]
        k_enc = (1 - k_s) * k_1 + k_s * g_1

        # Step.1 Global Graph Linking
        q_glob = self.glob_query_linear(q_enc)
        qk_logit = torch.matmul(q_glob, k_enc.transpose(1, 0))
        qk_r_matmul = torch.matmul(q_glob.unsqueeze(1), rel_k.transpose(-1, -2)).squeeze(1)
        glob_weight = (qk_logit + qk_r_matmul) / math.sqrt(self.d_model)
        glob_attn = F.softmax(self.activation_func(glob_weight), dim=-1)

        # Step.2 Local Graph Linking
        q_loc = self.loc_query_linear(q_enc)
        qnk_logit = torch.matmul(q_loc, k_enc.transpose(1, 0))
        qnk_r_matmul = torch.matmul(q_loc.unsqueeze(1), rel_k.transpose(-1, -2)).squeeze(1)
        loc_weight = (qnk_logit + qnk_r_matmul) / math.sqrt(self.d_model)
        loc_attn = self.activation_func(loc_weight)

        # Step.3 Dual-Graph Aggregation
        loc_weight_rep = loc_attn.repeat(1, k_num).view(q_num, k_num, k_num)
        zero_vec = -9e15 * torch.ones_like(loc_weight_rep)
        attn_adj_norm = F.softmax(torch.where(k_adj != 0, loc_weight_rep, zero_vec), dim=-1)
        k_loc_attn_val = torch.matmul(attn_adj_norm, k_enc)
        k_self_attn_val = k_enc.repeat(q_num, 1, 1)

        loc_gate = torch.sigmoid(self.loc_gate_linear(torch.cat([k_loc_attn_val, k_self_attn_val], dim=-1)))
        k_awr_enc = loc_gate * k_loc_attn_val + (1 - loc_gate) * k_self_attn_val

        k_matmul = torch.matmul(glob_attn.unsqueeze(1), k_awr_enc).squeeze(1)
        k_r_matmul = torch.matmul(glob_attn.unsqueeze(1), rel_v).squeeze(1)
        q_enc_new = k_matmul + k_r_matmul

        gate = torch.sigmoid(self.gate_linear(torch.cat([q_enc_new, q_enc], dim=-1)))
        h = gate * q_enc_new + (1 - gate) * q_enc

        return h
