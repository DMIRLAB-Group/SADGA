{
    logdir: "logdir/",
    model_config: "configs/sadga-bert-config.jsonnet",
    infer:{
        res_dir: "res",
        beam_size: 1,
        infer_name: "sadga_bert",
        pred_name: "predict_sql",
        section: "dev",
        start_step: 57100,
    },
    eval:{
        data_dir: 'dataset/',
        acc_res_name: "acc_res",
    },
}