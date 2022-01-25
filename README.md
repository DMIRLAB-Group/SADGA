# SADGA

The PyTorch implementation of paper [SADGA: Structure-Aware Dual Graph Aggregation Network for Text-to-SQL](https://arxiv.org/abs/2111.00653). (NeurIPS 2021)

If you use SADGA in your work, please cite it as follows:
``` bibtex
@article{cai2021sadga,
  title={SADGA: Structure-Aware Dual Graph Aggregation Network for Text-to-SQL},
  author={Cai, Ruichu and Yuan, Jinjie and Xu, Boyan and Hao, Zhifeng},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  year={2021}
}
```

## Usage

### Download dataset, third-party dependency and pretrained language model.

``` bash
mkdir -p dataset third_party plm
```

Download the dataset: [Spider](https://yale-lily.github.io/spider). Then unzip `spider.zip` into the directory `dataset`.

```
└── dataset
    ├── database
    │   ├── academic
    │   │   ├──academic.sqlite
    │   │   ├──schema.sql
    │   ├── ...
    ├── dev_gold.sql
    ├── dev.json
    ├── README.txt
    ├── tables.json
    ├── train_gold.sql
    ├── train_others.json
    └── train_spider.json
```

Download and unzip [Stanford CoreNLP](https://download.cs.stanford.edu/nlp/software/stanford-corenlp-full-2018-10-05.zip) to the directory `third_party`. Note that this repository requires a JVM to run it.

```
└── third_party
    └── stanford-corenlp-full-2018-10-05
        ├── ...
```

Regarding the implementation with the pretrained model GAP, download the pertained model from [pretrained-checkpoint](https://gap-text2sql-public.s3.amazonaws.com/checkpoint-artifacts/pretrained-checkpoint) into the directory `plm`.

```
└── plm
    └── gap
        └── pytorch_model.bin
```


### Create environment

We trained our models on one server with a single NVIDIA GTX 3090 GPU with 24GB GPU memory. In our experiments, we use **python 3.7**,  **torch 1.7.1** with **CUDA version 11.0**. We create conda environment `sadgasql`:

```bash
    conda create -n sadgasql python=3.7
    source activate sadgasql
    pip install torch==1.7.1+cu110 -f https://download.pytorch.org/whl/torch_stable.html
    pip install -r requirements.txt
    python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
```

### Run

All configs of the experiment and model  are in the files `sadga-glove-run.jsonnet`, `sadga-bert-run.jsonnet`, `sadga-gap-run.jsonnet`.

##### Step 1. Preprocess

```bash
    python run.py --mode preprocess --config sadga-[glove|bert|gap]-run.jsonnet
```

- Our preprocessed dataset can be downloaded [here](https://drive.google.com/file/d/1Bcn4N5GQEK3DPBpDCntayUbtGc0VNvwQ/view?usp=sharing)(1.6M).

##### Step 2. Training

```bash
    python run.py --mode train --config sadga-[glove|bert|gap]-run.jsonnet
```

- After the training, we can obtain some model-checkpoints in the directory `{logdir}/{model_name}/`, e.g., `logdir/sadga_glove_bs=20_lr=7.4e-04/model_checkpoint-00020100`.

##### Step 3. Inference

```bash
    python run.py --mode infer --config sadga-[glove|bert|gap]-run.jsonnet
```

- The inference phase aims to output the predicted SQL file `predict_sql_step{xxx}.txt`(the same input format as the official [Spider Evaluation](https://github.com/taoyds/spider)) in the directory `{logdir}/{model_name}/{res_dir}` for each saved models, e.g., `logdir/sadga_glove_bs=20_lr=7.4e-04/res/predict_sql_step20100.txt`.

##### Step 4. Eval

```bash
    python run.py --mode eval --config sadga-[glove|bert|gap]-run.jsonnet
```

- The Spider's official evaluation. We can get the final detailed accuracy result file `acc_res_step{xxx}.txt` for each saved models,  e.g., `logdir/sadga_glove_bs=20_lr=7.4e-04/res/acc_res_step20100.txt`, and the program can print the all inferred steps results as:

 ```
    STEP		ACCURACY
    10100	0.544
    11100	0.560
    ...
    40000	0.652
    Best Result: 
    38100	0.656
 ```

## Results

Our best trained checkpoints, predict_sql files and acc_res files can be downloaded in here:[[logdir.zip](https://drive.google.com/file/d/1Ip5_hsLb4gwoDbsAuStAQ7up5KtFLb9b/view?usp=sharing)] (GloVe), [[logdir.zip](https://drive.google.com/file/d/1Ip5_hsLb4gwoDbsAuStAQ7up5KtFLb9b/view?usp=sharing)] (Bert-large) and  [[logdir.zip](https://drive.google.com/file/d/1Ip5_hsLb4gwoDbsAuStAQ7up5KtFLb9b/view?usp=sharing)] (GAP).

|      Model       | Exact Match Acc (Dev) | Exact Match Acc (Test) |
| :--------------: | :------: | :------: |
| SADGA + GloVe |64.7 ( 65.6 this repo ) |   -   |
| SADGA + Bert-large |    71.6   |   66.7   |
| SADGA + GAP |    73.1   |   70.1   |

Detailed results can be found in the [paper](https://arxiv.org/abs/2111.00653).

## Acknowledgements

This implementation is based on the ACL 2020 paper [RAT-SQL: Relation-Aware Schema Encoding and Linking for Text-to-SQL Parsers](https://arxiv.org/abs/1911.04942) ([code](https://github.com/Microsoft/rat-sql)) . Thanks to the open-source project. We thank Tao Yu and Yusen Zhang for their evaluation of our work in the Spider Challenge. We also thank the anonymous reviewers for their helpful comments.

