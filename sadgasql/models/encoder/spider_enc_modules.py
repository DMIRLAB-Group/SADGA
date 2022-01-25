import itertools
import operator

import numpy as np
import torch
from torch import nn

from sadgasql.models.decoder import variational_lstm
from sadgasql.models.encoder import rat, sadga
from sadgasql.utils import batched_sequence


def clamp(value, abs_max):
    value = max(-abs_max, value)
    value = min(abs_max, value)
    return value


def get_attn_mask(seq_lengths):
    # Given seq_lengths like [3, 1, 2], this will produce
    # [[[1, 1, 1],
    #   [1, 1, 1],
    #   [1, 1, 1]],
    #  [[1, 0, 0],
    #   [0, 0, 0],
    #   [0, 0, 0]],
    #  [[1, 1, 0],
    #   [1, 1, 0],
    #   [0, 0, 0]]]
    # int(max(...)) so that it has type 'int instead of numpy.int64
    max_length, batch_size = int(max(seq_lengths)), len(seq_lengths)
    attn_mask = torch.LongTensor(batch_size, max_length, max_length).fill_(0)
    for batch_idx, seq_length in enumerate(seq_lengths):
        attn_mask[batch_idx, :seq_length, :seq_length] = 1
    return attn_mask


class LookupEmbeddings(torch.nn.Module):
    def __init__(self, device, vocab, embedder, emb_size, learnable_words=[]):
        super().__init__()
        self._device = device
        self.vocab = vocab
        self.embedder = embedder
        self.emb_size = emb_size

        self.embedding = torch.nn.Embedding(
            num_embeddings=len(self.vocab),
            embedding_dim=emb_size)
        if self.embedder:
            assert emb_size == self.embedder.dim

        # init embedding
        self.learnable_words = learnable_words
        init_embed_list = []
        for i, word in enumerate(self.vocab):
            if self.embedder.contains(word):
                init_embed_list.append( \
                    self.embedder.lookup(word))
            else:
                init_embed_list.append(self.embedding.weight[i])
        init_embed_weight = torch.stack(init_embed_list, 0)
        self.embedding.weight = nn.Parameter(init_embed_weight)

    def forward_unbatched(self, token_lists):
        # token_lists: list of list of lists
        # [batch, num descs, desc length]
        # - each list contains tokens
        # - each list corresponds to a column name, table name, etc.

        embs = []
        for tokens in token_lists:
            # token_indices shape: batch (=1) x length
            token_indices = torch.tensor(
                self.vocab.indices(tokens), device=self._device).unsqueeze(0)

            # emb shape: batch (=1) x length x word_emb_size
            emb = self.embedding(token_indices)

            # emb shape: desc length x batch (=1) x word_emb_size
            emb = emb.transpose(0, 1)
            embs.append(emb)

        # all_embs shape: sum of desc lengths x batch (=1) x word_emb_size
        all_embs = torch.cat(embs, dim=0)

        # boundaries shape: num of descs + 1
        # If desc lengths are [2, 3, 4],
        # then boundaries is [0, 2, 5, 9]
        boundaries = np.cumsum([0] + [emb.shape[0] for emb in embs])

        return all_embs, boundaries

    def _compute_boundaries(self, token_lists):
        # token_lists: list of list of lists
        # [batch, num descs, desc length]
        # - each list contains tokens
        # - each list corresponds to a column name, table name, etc.
        boundaries = [
            np.cumsum([0] + [len(token_list) for token_list in token_lists_for_item])
            for token_lists_for_item in token_lists]

        return boundaries

    def _embed_token(self, token, batch_idx):
        if token in self.learnable_words or not self.embedder.contains(token):
            return self.embedding.weight[self.vocab.index(token)]
        else:
            emb = self.embedder.lookup(token)
            return emb.to(self._device)

    def forward(self, token_lists):
        # token_lists: list of list of lists
        # [batch, num descs, desc length]
        # - each list contains tokens
        # - each list corresponds to a column name, table name, etc.
        # PackedSequencePlus, with shape: [batch, sum of desc lengths, emb_size]
        all_embs = batched_sequence.PackedSequencePlus.from_lists(
            lists=[
                [
                    token
                    for token_list in token_lists_for_item
                    for token in token_list
                ]
                for token_lists_for_item in token_lists
            ],
            item_shape=(self.emb_size,),
            device=self._device,
            item_to_tensor=self._embed_token)
        all_embs = all_embs.apply(lambda d: d.to(self._device))

        return all_embs, self._compute_boundaries(token_lists)

    def _embed_words_learned(self, token_lists):
        # token_lists: list of list of lists
        # [batch, num descs, desc length]
        # - each list contains tokens
        # - each list corresponds to a column name, table name, etc.

        # PackedSequencePlus, with shape: [batch, num descs * desc length (sum of desc lengths)]
        indices = batched_sequence.PackedSequencePlus.from_lists(
            lists=[
                [
                    token
                    for token_list in token_lists_for_item
                    for token in token_list
                ]
                for token_lists_for_item in token_lists
            ],
            item_shape=(1,),  # For compatibility with old PyTorch versions
            tensor_type=torch.LongTensor,
            item_to_tensor=lambda token, batch_idx, out: out.fill_(self.vocab.index(token))
        )
        indices = indices.apply(lambda d: d.to(self._device))
        # PackedSequencePlus, with shape: [batch, sum of desc lengths, emb_size]
        all_embs = indices.apply(lambda x: self.embedding(x.squeeze(-1)))

        return all_embs, self._compute_boundaries(token_lists)


class EmbLinear(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = torch.nn.Linear(input_size, output_size)

    def forward(self, input_):
        all_embs, boundaries = input_
        all_embs = all_embs.apply(lambda d: self.linear(d))
        return all_embs, boundaries


class BiLSTM(torch.nn.Module):
    def __init__(self, input_size, output_size, dropout, summarize, use_native=False):
        # input_size: dimensionality of input
        # output_size: dimensionality of output
        # dropout
        # summarize:
        # - True: return Tensor of 1 x batch x emb size 
        # - False: return Tensor of seq len x batch x emb size 
        super().__init__()

        if use_native:
            self.lstm = torch.nn.LSTM(
                input_size=input_size,
                hidden_size=output_size // 2,
                bidirectional=True,
                dropout=dropout)
            self.dropout = torch.nn.Dropout(dropout)
        else:
            self.lstm = variational_lstm.LSTM(
                input_size=input_size,
                hidden_size=int(output_size // 2),
                bidirectional=True,
                dropout=dropout)
        self.summarize = summarize
        self.use_native = use_native

    def forward_unbatched(self, input_):
        # all_embs shape: sum of desc lengths x batch (=1) x input_size
        all_embs, boundaries = input_

        new_boundaries = [0]
        outputs = []
        for left, right in zip(boundaries, boundaries[1:]):
            # state shape:
            # - h: num_layers (=1) * num_directions (=2) x batch (=1) x recurrent_size / 2
            # - c: num_layers (=1) * num_directions (=2) x batch (=1) x recurrent_size / 2
            # output shape: seq len x batch size x output_size
            if self.use_native:
                inp = self.dropout(all_embs[left:right])
                output, (h, c) = self.lstm(inp)
            else:
                output, (h, c) = self.lstm(all_embs[left:right])
            if self.summarize:
                seq_emb = torch.cat((h[0], h[1]), dim=-1).unsqueeze(0)
                new_boundaries.append(new_boundaries[-1] + 1)
            else:
                seq_emb = output
                new_boundaries.append(new_boundaries[-1] + output.shape[0])
            outputs.append(seq_emb)

        return torch.cat(outputs, dim=0), new_boundaries

    def forward(self, input_):
        # all_embs shape: PackedSequencePlus with shape [batch, sum of desc lengths, input_size]
        # boundaries: list of lists with shape [batch, num descs + 1]
        all_embs, boundaries = input_

        # List of the following:
        # (batch_idx, desc_idx, length)
        desc_lengths = []
        batch_desc_to_flat_map = {}
        for batch_idx, boundaries_for_item in enumerate(boundaries):
            for desc_idx, (left, right) in enumerate(zip(boundaries_for_item, boundaries_for_item[1:])):
                desc_lengths.append((batch_idx, desc_idx, right - left))
                batch_desc_to_flat_map[batch_idx, desc_idx] = len(batch_desc_to_flat_map)

        # Recreate PackedSequencePlus into shape
        # [batch * num descs, desc length, input_size]
        # with name `rearranged_all_embs`
        remapped_ps_indices = []

        def rearranged_all_embs_map_index(desc_lengths_idx, seq_idx):
            batch_idx, desc_idx, _ = desc_lengths[desc_lengths_idx]
            return batch_idx, boundaries[batch_idx][desc_idx] + seq_idx

        def rearranged_all_embs_gather_from_indices(indices):
            batch_indices, seq_indices = zip(*indices)
            remapped_ps_indices[:] = all_embs.raw_index(batch_indices, seq_indices)
            return all_embs.ps.data[torch.LongTensor(remapped_ps_indices)]

        rearranged_all_embs = batched_sequence.PackedSequencePlus.from_gather(
            lengths=[length for _, _, length in desc_lengths],
            map_index=rearranged_all_embs_map_index,
            gather_from_indices=rearranged_all_embs_gather_from_indices)
        rev_remapped_ps_indices = tuple(
            x[0] for x in sorted(
                enumerate(remapped_ps_indices), key=operator.itemgetter(1)))

        # output shape: PackedSequence, [batch * num_descs, desc length, output_size]
        # state shape:
        # - h: [num_layers (=1) * num_directions (=2), batch, output_size / 2]
        # - c: [num_layers (=1) * num_directions (=2), batch, output_size / 2]
        if self.use_native:
            rearranged_all_embs = rearranged_all_embs.apply(self.dropout)
        output, (h, c) = self.lstm(rearranged_all_embs.ps)
        if self.summarize:
            # h shape: [batch * num descs, output_size]
            h = torch.cat((h[0], h[1]), dim=-1)

            # new_all_embs: PackedSequencePlus, [batch, num descs, input_size]
            new_all_embs = batched_sequence.PackedSequencePlus.from_gather(
                lengths=[len(boundaries_for_item) - 1 for boundaries_for_item in boundaries],
                map_index=lambda batch_idx, desc_idx: rearranged_all_embs.sort_to_orig[
                    batch_desc_to_flat_map[batch_idx, desc_idx]],
                gather_from_indices=lambda indices: h[torch.LongTensor(indices)])

            new_boundaries = [
                list(range(len(boundaries_for_item)))
                for boundaries_for_item in boundaries
            ]
        else:
            new_all_embs = all_embs.apply(
                lambda _: output.data[torch.LongTensor(rev_remapped_ps_indices)])
            new_boundaries = boundaries

        return new_all_embs, new_boundaries


class StructureAwareGraphAggrUpdate(torch.nn.Module):

    def __init__(self,
                 device,
                 hidden_size,
                 sadga_num_layers,
                 rat_num_layers,
                 num_heads,
                 ggnn_num_layers,
                 word_max_dist=2,
                 ff_size=None,
                 dropout=0.1,
                 sadga_dropout=0.5,
                 activation_func='tanh'):
        super().__init__()
        self._device = device
        self.ggnn_edge_type = 3

        self.relation_ids = {}
        def add_relation(name):
            self.relation_ids[name] = len(self.relation_ids)

        def add_rel_dist(name, max_dist):
            for i in range(1, max_dist + 1):
                add_relation((name, i))

        # Pre-defined Relations
        add_relation('default')
        add_rel_dist('qq_dist', word_max_dist)
        add_relation('cc_foreign_key')
        add_relation('cc_table_match')
        add_relation('ct_foreign_key')
        add_relation('ct_primary_key')
        add_relation('ct_table_match')
        add_relation('tt_foreign_key')
        add_relation('qcCEM')
        add_relation('qtTEM')
        add_relation('qcCPM')
        add_relation('qtTPM')
        add_relation("qcNUMBER")
        add_relation("qcTIME")
        add_relation("qcCELLMATCH")

        if ff_size is None:
            ff_size = hidden_size * 4

        self.sadga_encoder = sadga.Encoder(
            lambda: sadga.EncoderLayer(
                hidden_size,
                sadga.SadgaLayer(
                    hidden_size,
                    ggnn_num_layers,
                    self.ggnn_edge_type,
                    len(self.relation_ids),
                    self._device,
                    activation_func=activation_func,
                    dropout=sadga_dropout),
                sadga.PositionwiseFeedForward(
                    hidden_size,
                    ff_size,
                    dropout),
                dropout),
            hidden_size,
            sadga_num_layers)

        self.rat_encoder = rat.Encoder(
            lambda: rat.EncoderLayer(
                hidden_size,
                rat.MultiHeadedAttentionWithRelations(
                    num_heads,
                    hidden_size,
                    dropout),
                rat.PositionwiseFeedForward(
                    hidden_size,
                    ff_size,
                    dropout),
                len(self.relation_ids),
                dropout),
            hidden_size,
            rat_num_layers)

        self.align_attn = rat.PointerWithRelations(hidden_size, len(self.relation_ids), dropout)

    def forward_unbatched(self, desc, q_enc, c_enc, c_boundaries, t_enc, t_boundaries):
        enc = torch.cat((q_enc, c_enc, t_enc), dim=0)
        enc = enc.transpose(0, 1)

        relations = self.compute_relations(
            desc,
            enc_length=enc.shape[1],
            q_enc_length=q_enc.shape[0],
            c_enc_length=c_enc.shape[0],
            c_boundaries=c_boundaries,
            t_boundaries=t_boundaries)

        relations_t = torch.LongTensor(relations).to(self._device)
        sadga_updated = self.sadga_encoder(enc, relations, q_len=q_enc.size(0), desc=desc)
        enc_new = self.rat_encoder(sadga_updated, relations_t, mask=None)

        c_base = q_enc.shape[0]
        t_base = q_enc.shape[0] + c_enc.shape[0]
        q_enc_new = enc_new[:, :c_base]
        c_enc_new = enc_new[:, c_base:t_base]
        t_enc_new = enc_new[:, t_base:]

        m2c_align_mat = self.align_attn(enc_new, enc_new[:, c_base:t_base], \
                                        enc_new[:, c_base:t_base], relations_t[:, c_base:t_base])
        m2t_align_mat = self.align_attn(enc_new, enc_new[:, t_base:], \
                                        enc_new[:, t_base:], relations_t[:, t_base:])
        return q_enc_new, c_enc_new, t_enc_new, (m2c_align_mat, m2t_align_mat)

    def compute_relations(self, desc, enc_length, q_enc_length, c_enc_length, c_boundaries, t_boundaries):
        sc_link = desc.get('sc_link', {'q_col_match': {}, 'q_tab_match': {}})
        cv_link = desc.get('cv_link', {'num_date_match': {}, 'cell_match': {}})
        loc_types = {}
        for i in range(q_enc_length):
            loc_types[i] = ('question',)

        c_base = q_enc_length
        for c_id, (c_start, c_end) in enumerate(zip(c_boundaries, c_boundaries[1:])):
            for i in range(c_start + c_base, c_end + c_base):
                loc_types[i] = ('column', c_id)
        t_base = q_enc_length + c_enc_length
        for t_id, (t_start, t_end) in enumerate(zip(t_boundaries, t_boundaries[1:])):
            for i in range(t_start + t_base, t_end + t_base):
                loc_types[i] = ('table', t_id)

        relations = np.zeros((enc_length, enc_length), dtype=np.int64)

        for i, j in itertools.product(range(enc_length), repeat=2):
            def set_relation(name):
                relations[i, j] = self.relation_ids[name]

            i_type, j_type = loc_types[i], loc_types[j]
            if i_type[0] == 'question':
                if j_type[0] == 'question':
                    if 0 < abs(j - i) < 3:
                        set_relation(('qq_dist', abs(j - i)))
                elif j_type[0] == 'column':
                    j_real = j - c_base
                    if f"{i},{j_real}" in sc_link["q_col_match"]:
                        set_relation("qc" + sc_link["q_col_match"][f"{i},{j_real}"])
                    elif f"{i},{j_real}" in cv_link["cell_match"]:
                        set_relation("qc" + cv_link["cell_match"][f"{i},{j_real}"])
                    elif f"{i},{j_real}" in cv_link["num_date_match"]:
                        set_relation("qc" + cv_link["num_date_match"][f"{i},{j_real}"])

                elif j_type[0] == 'table':
                    j_real = j - t_base
                    if f"{i},{j_real}" in sc_link["q_tab_match"]:
                        set_relation("qt" + sc_link["q_tab_match"][f"{i},{j_real}"])

            elif i_type[0] == 'column':
                if j_type[0] == 'question':
                    i_real = i - c_base
                    if f"{j},{i_real}" in sc_link["q_col_match"]:
                        set_relation("qc" + sc_link["q_col_match"][f"{j},{i_real}"])
                    elif f"{j},{i_real}" in cv_link["cell_match"]:
                        set_relation("qc" + cv_link["cell_match"][f"{j},{i_real}"])
                    elif f"{j},{i_real}" in cv_link["num_date_match"]:
                        set_relation("qc" + cv_link["num_date_match"][f"{j},{i_real}"])

                elif j_type[0] == 'column':
                    col1, col2 = i_type[1], j_type[1]
                    if desc['foreign_keys'].get(str(col1)) == col2:
                        set_relation('cc_foreign_key')
                    if desc['foreign_keys'].get(str(col2)) == col1:
                        set_relation('cc_foreign_key')
                    if (desc['column_to_table'][str(col1)] == desc['column_to_table'][str(col2)]):
                        set_relation('cc_table_match')

                elif j_type[0] == 'table':
                    col, table = i_type[1], j_type[1]
                    if self.match_foreign_key(desc, col, table):
                        set_relation('ct_foreign_key')
                    col_table = desc['column_to_table'][str(col)]
                    if col_table == table:
                        if col in desc['primary_keys']:
                            set_relation('ct_primary_key')
                        else:
                            set_relation('ct_table_match')

            elif i_type[0] == 'table':
                if j_type[0] == 'question':
                    i_real = i - t_base
                    if f"{j},{i_real}" in sc_link["q_tab_match"]:
                        set_relation("qt" + sc_link["q_tab_match"][f"{j},{i_real}"])

                elif j_type[0] == 'column':
                    table, col = i_type[1], j_type[1]

                    if self.match_foreign_key(desc, col, table):
                        set_relation('ct_foreign_key')
                    col_table = desc['column_to_table'][str(col)]
                    if col_table == table:
                        if col in desc['primary_keys']:
                            set_relation('ct_primary_key')
                        else:
                            set_relation('ct_table_match')

                elif j_type[0] == 'table':
                    table1, table2 = i_type[1], j_type[1]
                    forward = table2 in desc['foreign_keys_tables'].get(str(table1), ())
                    backward = table1 in desc['foreign_keys_tables'].get(str(table2), ())
                    if forward or backward:
                        set_relation('tt_foreign_key')

        return relations

    @classmethod
    def match_foreign_key(cls, desc, col, table):
        foreign_key_for = desc['foreign_keys'].get(str(col))
        if foreign_key_for is None:
            return False
        foreign_table = desc['column_to_table'][str(foreign_key_for)]
        return desc['column_to_table'][str(col)] == foreign_table
