{
    local batch_size = 6,
    local num_batch_accumulated = 4,
    local clip_grad = 1,
    local max_steps = 81000,
    local lr = 2.44e-4,
    local bert_lr = 3e-6,
    local _data_path = 'dataset/',
    local bert_version = 'bert-large-uncased-whole-word-masking',

    model_name: 'sadga_bert_bs=%(bs)d_lr=%(lr)s_blr=%(blr)s' % ({
        bs: batch_size,
        lr: '%0.1e' % lr,
        blr: '%0.1e' % bert_lr,
    }),

    data: {
        train: {
            name: 'spider',
            paths: [
              _data_path + 'train_%s.json' % [s]
              for s in ['spider', 'others']
            ],
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
            name: 'Encoder4Bert',
            bert_token_type: true,
            bert_version: bert_version,
            summarize_header: 'avg',
            use_column_type: false,
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
        },
        encoder_preproc: {
            bert_version: bert_version,
            include_table_name_in_column: false,
            db_path: _data_path + "database",
            compute_sc_link: true,
            compute_cv_link: true,
            fix_issue_16_primary_keys: true,
            save_path: "preprocess_data/bert/",
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
            save_path: "preprocess_data/bert/",
        },
        decoder: {
            name: 'Decoder',
            dropout: 0.20687225956012834,
            desc_attn: 'mha',
            recurrent_size : 512,
            enc_recurrent_size: 1024,
            loss_type: "label_smooth",
            use_align_mat: true,
            use_align_loss: true,
        },
    },

    train: {
        batch_size: batch_size,
        num_batch_accumulated: num_batch_accumulated,
        clip_grad: clip_grad,
        model_seed: 1,
        data_seed:  1,
        init_seed:  1,
        keep_every_n: 1000,
        save_every_n: 100,
        report_every_n: 10,
        max_steps: max_steps,
    },

    optimizer: {
        bert_lr: 0.0,
        lr: 0.0,
        name: 'bertAdamw',
    },

    lr_scheduler: {
        name: 'bert_warmup_polynomial_group',
        start_lrs: [lr, bert_lr],
        num_warmup_steps: $.train.max_steps / 8,
        start_lr: 1e-3,
        end_lr: 0.0,
        decay_steps: $.train.max_steps - self.num_warmup_steps,
        power: 0.5,
    },

    log: {
        reopen_to_flush: true,
    }

}
