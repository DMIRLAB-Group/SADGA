{
    local lr = 0.000743552663260837,
    local end_lr = 0,
    local bs = 20,
    local _data_path = 'dataset/',
    local lr_s = '%0.1e' % lr,
    local end_lr_s = '0e0',

    model_name: 'sadga_glove_bs=%(bs)d_lr=%(lr)s' % ({
        bs: bs,
        lr: lr_s,
    }),

    data: {
        train: {
            name: 'spider',
            paths: [
              _data_path + 'train_%s.json' % [s]
              for s in ['spider', 'others']],
            tables_paths: [
              _data_path + 'tables.json',
            ],
            db_path: _data_path + 'database',
        },
        dev: {
            name: 'spider',
            paths: [_data_path + 'dev.json'],
            tables_paths: [_data_path + 'tables.json'],
            db_path: _data_path + 'database',
        },
    },

    model: {
        name: 'EncDec',
        encoder: {
            name: 'Encoder',
            hidden_size: 256,
            word_emb_size: 300,
            question_encoder: ['emb', 'bilstm'],
            column_encoder: ['emb', 'bilstm-summarize'],
            table_encoder: ['emb', 'bilstm-summarize'],
            update_config:  {
                name: 'sadga',
                sadga_num_layers: 3,
                rat_num_layers: 4,
                num_heads: 8,
                ggnn_num_layers: 2,
                word_max_dist: 2,
                hidden_size: 256,
                dropout: 0.1,
                sadga_dropout: 0.5,
            },
            top_k_learnable: 50,
            dropout: 0.2,
        },
        encoder_preproc: {
            word_emb: {
                name: 'glove',
                kind: '42B',
                lemmatize: true,
            },
            include_table_name_in_column: false,
            min_freq: 4,
            max_count: 5000,
            db_path: _data_path + "database",
            compute_sc_link: true,
            compute_cv_link: true,
            fix_issue_16_primary_keys: true,
            count_tokens_in_word_emb_for_vocab: true,
            save_path: "preprocess_data",
        },
        decoder_preproc: {
            grammar: {
                name: 'spider',
                output_from: true,
                use_table_pointer: true,
                include_literals: false,
                end_with_from: true,
                infer_from_conditions: true,
                clause_order: null,
                factorize_sketch: 2,
            },
            use_seq_elem_rules: true,
            min_freq: 4,
            max_count: 5000,
            save_path: "preprocess_data",
        },
        decoder: {
            name: 'Decoder',
            dropout: 0.20687225956012834,
            desc_attn: 'mha',
            recurrent_size : 512,
            enc_recurrent_size: 256,
            loss_type: "softmax",
            use_align_mat: true,
            use_align_loss: true,
            enumerate_order: false,
        },
    },

    train+: {
        batch_size: bs,
        num_batch_accumulated: 1,
        clip_grad: null,
        model_seed: 0,
        data_seed:  0,
        init_seed:  0,
        keep_every_n: 1000,
        save_every_n: 100,
        report_every_n: 10,
        max_steps: 40000,
    },

    optimizer: {
        name: 'adam',
        lr: 0.0,
    },

    lr_scheduler: {
        name: 'warmup_polynomial',
        num_warmup_steps: $.train.max_steps / 20,
        start_lr: lr,
        end_lr: end_lr,
        decay_steps: $.train.max_steps - self.num_warmup_steps,
        power: 0.5,
    }

}
