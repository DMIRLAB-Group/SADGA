import collections
import itertools
import json
import os
import attr
import torch
from stanfordcorenlp import StanfordCoreNLP
from sadgasql.models import abstract_preproc
from sadgasql.models.encoder import spider_enc_modules
from sadgasql.utils import registry
from sadgasql.utils import serialization
from sadgasql.utils import vocab
from sadgasql.utils.spider_match_utils import (
    compute_schema_linking,
    compute_cell_value_linking
)


@attr.s
class SpiderEncoderState:
    state = attr.ib()
    memory = attr.ib()
    question_memory = attr.ib()
    schema_memory = attr.ib()
    words = attr.ib()

    pointer_memories = attr.ib()
    pointer_maps = attr.ib()

    m2c_align_mat = attr.ib()
    m2t_align_mat = attr.ib()

    def find_word_occurrences(self, word):
        return [i for i, w in enumerate(self.words) if w == word]


@attr.s
class PreprocessedSchema:
    column_names = attr.ib(factory=list)
    table_names = attr.ib(factory=list)
    table_bounds = attr.ib(factory=list)
    column_to_table = attr.ib(factory=dict)
    table_to_columns = attr.ib(factory=dict)
    foreign_keys = attr.ib(factory=dict)
    foreign_keys_tables = attr.ib(factory=lambda: collections.defaultdict(set))
    primary_keys = attr.ib(factory=list)

    # only for bert version
    normalized_column_names = attr.ib(factory=list)
    normalized_table_names = attr.ib(factory=list)


def preprocess_schema_uncached(schema,
                               tokenize_func,
                               include_table_name_in_column,
                               fix_issue_16_primary_keys,
                               bert=False):
    """If it's bert, we also cache the normalized version of 
    question/column/table for schema linking"""
    r = PreprocessedSchema()

    last_table_id = None
    for i, column in enumerate(schema.columns):
        col_toks = tokenize_func(
            column.name, column.unsplit_name)

        # assert column.type in ["text", "number", "time", "boolean", "others"]
        type_tok = f'<type: {column.type}>'
        column_name = [type_tok] + col_toks

        if include_table_name_in_column:
            if column.table is None:
                table_name = ['<any-table>']
            else:
                table_name = tokenize_func(
                    column.table.name, column.table.unsplit_name)
            column_name += ['<table-sep>'] + table_name
        r.column_names.append(column_name)

        table_id = None if column.table is None else column.table.id
        r.column_to_table[str(i)] = table_id
        if table_id is not None:
            columns = r.table_to_columns.setdefault(str(table_id), [])
            columns.append(i)
        if last_table_id != table_id:
            r.table_bounds.append(i)
            last_table_id = table_id

        if column.foreign_key_for is not None:
            r.foreign_keys[str(column.id)] = column.foreign_key_for.id
            r.foreign_keys_tables[str(column.table.id)].add(column.foreign_key_for.table.id)

    r.table_bounds.append(len(schema.columns))
    assert len(r.table_bounds) == len(schema.tables) + 1

    for i, table in enumerate(schema.tables):
        table_toks = tokenize_func(
            table.name, table.unsplit_name)
        r.table_names.append(table_toks)
    last_table = schema.tables[-1]

    r.foreign_keys_tables = serialization.to_dict_with_sorted_values(r.foreign_keys_tables)
    r.primary_keys = [
        column.id
        for table in schema.tables
        for column in table.primary_keys
    ] if fix_issue_16_primary_keys else [
        column.id
        for column in last_table.primary_keys
        for table in schema.tables
    ]

    return r


class EncoderPreproc(abstract_preproc.AbstractPreproc):

    def __init__(
            self,
            save_path,
            min_freq=3,
            max_count=5000,
            include_table_name_in_column=True,
            word_emb=None,
            count_tokens_in_word_emb_for_vocab=False,
            fix_issue_16_primary_keys=False,
            compute_sc_link=False,
            compute_cv_link=False,
            db_path=None):
        if word_emb is None:
            self.word_emb = None
        else:
            self.word_emb = registry.construct('word_emb', word_emb)

        self.data_dir = os.path.join(save_path, 'enc')
        self.include_table_name_in_column = include_table_name_in_column
        self.count_tokens_in_word_emb_for_vocab = count_tokens_in_word_emb_for_vocab
        self.fix_issue_16_primary_keys = fix_issue_16_primary_keys
        self.compute_sc_link = compute_sc_link
        self.compute_cv_link = compute_cv_link
        self.texts = collections.defaultdict(list)
        self.db_path = db_path

        self.vocab_builder = vocab.VocabBuilder(min_freq, max_count)
        self.vocab_path = os.path.join(save_path, 'enc_vocab.json')
        self.vocab_word_freq_path = os.path.join(save_path, 'enc_word_freq.json')
        self.vocab = None
        self.counted_db_ids = set()
        self.preprocessed_schemas = {}

        self.stanfordcore_nlp = StanfordCoreNLP(os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__),
                    '../../../third_party/stanford-corenlp-full-2018-10-05')))

    def validate_item(self, item, section):
        return True, None

    def add_item(self, item, section, validation_info):
        preprocessed = self.preprocess_item(item, validation_info)
        self.texts[section].append(preprocessed)

        if section == 'train':
            if item.schema.db_id in self.counted_db_ids:
                to_count = preprocessed['question']
            else:
                self.counted_db_ids.add(item.schema.db_id)
                to_count = itertools.chain(
                    preprocessed['question'],
                    *preprocessed['columns'],
                    *preprocessed['tables'])

            for token in to_count:
                count_token = (
                        self.word_emb is None or
                        self.count_tokens_in_word_emb_for_vocab or
                        self.word_emb.lookup(token) is None)
                if count_token:
                    self.vocab_builder.add_word(token)

    def clear_items(self):
        self.texts = collections.defaultdict(list)

    def preprocess_item(self, item, validation_info):
        question, question_for_copying = self._tokenize_for_copying(item.text, item.orig['question'])
        preproc_schema = self._preprocess_schema(item.schema)
        if self.compute_sc_link:
            assert preproc_schema.column_names[0][0].startswith("<type:")
            column_names_without_types = [col[1:] for col in preproc_schema.column_names]
            sc_link = compute_schema_linking(question, column_names_without_types, preproc_schema.table_names)
        else:
            sc_link = {"q_col_match": {}, "q_tab_match": {}}

        if self.compute_cv_link:
            cv_link = compute_cell_value_linking(question, item.schema)
        else:
            cv_link = {"num_date_match": {}, "cell_match": {}}

        dependency_tree = self.stanfordcore_nlp.dependency_parse(item.orig['question'])

        word_adj_tuple = []
        start_index = 0
        for index, tuple in enumerate(dependency_tree):
            if tuple[0] == 'ROOT':
                start_index = index
                continue
            word_adj_tuple.append((tuple[1] + start_index - 1, tuple[2] + start_index - 1))

        return {
            'raw_question': item.orig['question'],
            'question': question,
            'question_for_copying': question_for_copying,
            'db_id': item.schema.db_id,
            'sc_link': sc_link,
            'cv_link': cv_link,
            'columns': preproc_schema.column_names,
            'tables': preproc_schema.table_names,
            'table_bounds': preproc_schema.table_bounds,
            'column_to_table': preproc_schema.column_to_table,
            'table_to_columns': preproc_schema.table_to_columns,
            'foreign_keys': preproc_schema.foreign_keys,
            'foreign_keys_tables': preproc_schema.foreign_keys_tables,
            'primary_keys': preproc_schema.primary_keys,
            'word_adj_tuple': word_adj_tuple,
        }

    def _preprocess_schema(self, schema):
        if schema.db_id in self.preprocessed_schemas:
            return self.preprocessed_schemas[schema.db_id]
        result = preprocess_schema_uncached(schema, self._tokenize,
                                            self.include_table_name_in_column, self.fix_issue_16_primary_keys)
        self.preprocessed_schemas[schema.db_id] = result
        return result

    def _tokenize(self, presplit, unsplit):
        if self.word_emb:
            return self.word_emb.tokenize(unsplit)
        return presplit

    def _tokenize_for_copying(self, presplit, unsplit):
        if self.word_emb:
            return self.word_emb.tokenize_for_copying(unsplit)
        return presplit, presplit

    def save(self):
        os.makedirs(self.data_dir, exist_ok=True)
        self.vocab = self.vocab_builder.finish()
        print(f"{len(self.vocab)} words in vocab")
        self.vocab.save(self.vocab_path)
        self.vocab_builder.save(self.vocab_word_freq_path)

        for section, texts in self.texts.items():
            with open(os.path.join(self.data_dir, section + '.jsonl'), 'w') as f:
                for text in texts:
                    f.write(json.dumps(text) + '\n')

    def load(self):
        self.vocab = vocab.Vocab.load(self.vocab_path)
        self.vocab_builder.load(self.vocab_word_freq_path)

    def dataset(self, section):
        return [
            json.loads(line)
            for line in open(os.path.join(self.data_dir, section + '.jsonl'))]


@registry.register('encoder', 'Encoder')
class Encoder(torch.nn.Module):
    batched = True
    Preproc = EncoderPreproc

    def __init__(
            self,
            device,
            preproc,
            word_emb_size=300,
            hidden_size=256,
            dropout=0.,
            question_encoder=('emb', 'bilstm'),
            column_encoder=('emb', 'bilstm'),
            table_encoder=('emb', 'bilstm'),
            update_config={},
            include_in_memory=('question', 'column', 'table'),
            top_k_learnable=0):
        super().__init__()
        self._device = device
        self.preproc = preproc

        self.vocab = preproc.vocab
        self.word_emb_size = word_emb_size
        self.hidden_size = hidden_size
        assert self.hidden_size % 2 == 0
        word_freq = self.preproc.vocab_builder.word_freq
        top_k_words = set([_a[0] for _a in word_freq.most_common(top_k_learnable)])
        self.learnable_words = top_k_words

        self.include_in_memory = set(include_in_memory)
        self.dropout = dropout

        self.question_encoder = self._build_modules(question_encoder)
        self.column_encoder = self._build_modules(column_encoder)
        self.table_encoder = self._build_modules(table_encoder)

        self.encs_update = registry.instantiate(
            spider_enc_modules.StructureAwareGraphAggrUpdate,
            update_config,
            unused_keys={"name"},
            device=self._device,
            hidden_size=hidden_size,
        )

    def _build_modules(self, module_types):
        module_builder = {
            'emb': lambda: spider_enc_modules.LookupEmbeddings(
                self._device,
                self.vocab,
                self.preproc.word_emb,
                self.word_emb_size,
                self.learnable_words),
            'bilstm': lambda: spider_enc_modules.BiLSTM(
                input_size=self.word_emb_size,
                output_size=self.hidden_size,
                dropout=self.dropout,
                summarize=False),
            'bilstm-summarize': lambda: spider_enc_modules.BiLSTM(
                input_size=self.word_emb_size,
                output_size=self.hidden_size,
                dropout=self.dropout,
                summarize=True),
        }

        modules = []
        for module_type in module_types:
            modules.append(module_builder[module_type]())
        return torch.nn.Sequential(*modules)

    def forward(self, descs):
        qs = [[desc['question']] for desc in descs]
        q_enc, _ = self.question_encoder(qs)

        c_enc, c_boundaries = self.column_encoder([desc['columns'] for desc in descs])
        column_pointer_maps = [
            {
                i: list(range(left, right))
                for i, (left, right) in enumerate(zip(c_boundaries_for_item, c_boundaries_for_item[1:]))
            }
            for batch_idx, c_boundaries_for_item in enumerate(c_boundaries)
        ]

        t_enc, t_boundaries = self.table_encoder([desc['tables'] for desc in descs])
        table_pointer_maps = [
            {
                i: list(range(left, right))
                for i, (left, right) in enumerate(zip(t_boundaries_for_item, t_boundaries_for_item[1:]))
            }
            for batch_idx, (desc, t_boundaries_for_item) in enumerate(zip(descs, t_boundaries))
        ]

        result = []
        for batch_idx, desc in enumerate(descs):
            q_enc_new_item, c_enc_new_item, t_enc_new_item, align_mat_item = \
                self.encs_update.forward_unbatched(
                    desc,
                    q_enc.select(batch_idx).unsqueeze(1),
                    c_enc.select(batch_idx).unsqueeze(1),
                    c_boundaries[batch_idx],
                    t_enc.select(batch_idx).unsqueeze(1),
                    t_boundaries[batch_idx])

            memory = []
            words_for_copying = []
            if 'question' in self.include_in_memory:
                memory.append(q_enc_new_item)
                if 'question_for_copying' in desc:
                    assert q_enc_new_item.shape[1] == len(desc['question_for_copying'])
                    words_for_copying += desc['question_for_copying']
                else:
                    words_for_copying += [''] * q_enc_new_item.shape[1]
            if 'column' in self.include_in_memory:
                memory.append(c_enc_new_item)
                words_for_copying += [''] * c_enc_new_item.shape[1]
            if 'table' in self.include_in_memory:
                memory.append(t_enc_new_item)
                words_for_copying += [''] * t_enc_new_item.shape[1]
            memory = torch.cat(memory, dim=1)

            result.append(SpiderEncoderState(
                state=None,
                memory=memory,
                question_memory=q_enc_new_item,
                schema_memory=torch.cat((c_enc_new_item, t_enc_new_item), dim=1),
                words=words_for_copying,
                pointer_memories={
                    'column': c_enc_new_item,
                    'table': torch.cat((c_enc_new_item, t_enc_new_item), dim=1),
                },
                pointer_maps={
                    'column': column_pointer_maps[batch_idx],
                    'table': table_pointer_maps[batch_idx],
                },
                m2c_align_mat=align_mat_item[0],
                m2t_align_mat=align_mat_item[1],
            ))
        return result