#!/usr/bin/env python

import argparse
import json
import os

import _jsonnet
import attr
from sadgasql.commands import preprocess, train, infer, eval

@attr.s
class PreprocessConfig:
    config = attr.ib()

@attr.s
class TrainConfig:
    config = attr.ib()
    logdir = attr.ib()
    exp_config = attr.ib()

@attr.s
class InferConfig:
    config = attr.ib()
    logdir = attr.ib()
    section = attr.ib()
    beam_size = attr.ib()
    res_dir = attr.ib()
    infer_name = attr.ib()
    pred_name = attr.ib()
    step = attr.ib()
    mode = attr.ib(default="infer")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', help="preprocess/train/infer/eval")
    parser.add_argument('--config')
    args = parser.parse_args()

    exp_config = json.loads(_jsonnet.evaluate_file(args.config))
    model_config_file = exp_config["model_config"]
    config = json.loads(_jsonnet.evaluate_file(model_config_file))
    logdir = os.path.join(exp_config['logdir'], config['model_name'])

    if args.mode == "preprocess":
        preprocess_config = PreprocessConfig(model_config_file)
        preprocess.main(preprocess_config)
    elif args.mode == "train":
        train_config = TrainConfig(model_config_file, logdir, exp_config)
        train.main(train_config)
    elif args.mode == "infer":
        max_steps = config['train']['max_steps']
        keep_every_n = config['train']['keep_every_n']
        res_dir = os.path.join(logdir, exp_config['infer']['res_dir'])
        start_step = exp_config['infer']['start_step'] - keep_every_n
        for root, dirs, files in os.walk(res_dir):
            for f in files:
                file_ext = f.split('.')
                if file_ext[-1] == 'infer':
                    step = int(file_ext[0][file_ext[0].index('step') + 4:])
                    start_step = max(step, start_step)
        infer_steps = [x for x in range(int(start_step + keep_every_n), max_steps + 1, keep_every_n)] + [max_steps]
        for step in infer_steps:
            infer_config = InferConfig(
                model_config_file,
                logdir,
                exp_config['infer']["section"],
                exp_config['infer']["beam_size"],
                res_dir,
                exp_config['infer']['infer_name'],
                exp_config['infer']['pred_name'],
                step)
            infer.main(infer_config)
    elif args.mode == "eval":
        res_dir = os.path.join(logdir, exp_config['infer']['res_dir'])
        eval.main(data_dir=exp_config['eval']['data_dir'],
                  res_dir=res_dir,
                  pred_name=exp_config['infer']['pred_name'],
                  acc_res_name=exp_config['eval']['acc_res_name'])


if __name__ == "__main__":
    main()