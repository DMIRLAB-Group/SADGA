import json
import os
import sys

import _jsonnet
import torch
import tqdm

# These imports are needed for registry.lookup
# noinspection PyUnresolvedReferences
from sadgasql import beam_search
# noinspection PyUnresolvedReferences
from sadgasql import datasets
# noinspection PyUnresolvedReferences
from sadgasql import grammars
# noinspection PyUnresolvedReferences
from sadgasql import models
# noinspection PyUnresolvedReferences
from sadgasql import optimizers
# noinspection PyUnresolvedReferences
from sadgasql.datasets import spider
from sadgasql.models.decoder import spider_beam_search
from sadgasql.utils import registry
from sadgasql.utils import saver as saver_mod


class Inferer:
    def __init__(self, config):
        self.config = config
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
            torch.set_num_threads(1)

        self.model_preproc = registry.instantiate(
            registry.lookup('model', config['model']).Preproc,
            config['model'],
            unused_keys=('name',))
        self.model_preproc.load()

    def load_model(self, logdir, step):
        '''Load a model (identified by the config used for construction) and return it'''
        model = registry.construct('model', self.config['model'], preproc=self.model_preproc, device=self.device, unused_keys=('decoder_preproc', 'encoder_preproc'))
        model.to(self.device)
        model.eval()

        saver = saver_mod.Saver({"model": model})
        last_step = saver.restore(logdir, step=step, map_location=self.device, item_keys=["model"])
        if not last_step:
            raise Exception(f"Attempting to infer on untrained model in {logdir}, step={step}")
        return model

    def infer(self, model, output_path, args):
        output = open(output_path, 'w')

        with torch.no_grad():
            config = self.config['data'][args.section]
            orig_data = datasets.spider.SpiderDataset(paths=config['paths'], tables_paths=config['tables_paths'],
                                                      db_path=config['db_path'])
            preproc_data = self.model_preproc.dataset(args.section)
            assert len(orig_data) == len(preproc_data)
            self._inner_infer(model, args.beam_size, orig_data, preproc_data, output)

    def _inner_infer(self, model, beam_size, sliced_orig_data, sliced_preproc_data, output,
                     use_heuristic=True, output_history=False):
        for i, (orig_item, preproc_item) in enumerate(
                tqdm.tqdm(zip(sliced_orig_data, sliced_preproc_data),
                          total=len(sliced_orig_data))):
            decoded = self._infer_one(model, orig_item, preproc_item, beam_size, output_history, use_heuristic)
            output.write(
                json.dumps({
                    'index': i,
                    'beams': decoded,
                }) + '\n')
            output.flush()

    def _infer_one(self, model, data_item, preproc_item, beam_size, output_history=False, use_heuristic=True):
        beams = spider_beam_search.beam_search_with_heuristics(
            model, data_item, preproc_item, beam_size=beam_size, max_steps=1000, from_cond=False)
        decoded = []
        for beam in beams:
            model_output, inferred_code = beam.inference_state.finalize()

            decoded.append({
                'orig_question': data_item.orig["question"],
                'model_output': model_output,
                'inferred_code': inferred_code,
                'score': beam.score,
                **({
                       'choice_history': beam.choice_history,
                       'score_history': beam.score_history,
                   } if output_history else {})})
        return decoded


def main(args):
    config = json.loads(_jsonnet.evaluate_file(args.config))

    infer_path = f"{args.res_dir}/{args.infer_name}_step{args.step}.infer"
    os.makedirs(os.path.dirname(infer_path), exist_ok=True)
    if os.path.exists(infer_path):
        print(f'Output file {infer_path} already exists')
        sys.exit(1)

    inferer = Inferer(config)
    model = inferer.load_model(args.logdir, args.step)
    inferer.infer(model, infer_path, args)

    # For official.
    # infer -> txt (the form of official evaluation)
    data_config = config['data'][args.section]
    data = spider.SpiderDataset(paths=data_config['paths'], tables_paths=data_config['tables_paths'],
                                db_path=data_config['db_path'])
    pred_path = f"{args.res_dir}/{args.pred_name}_step{args.step}.txt"
    with open(pred_path, "w") as fw:
        with open(infer_path) as f1:
            for index, line1 in enumerate(f1.readlines()):
                infer_1 = json.loads(line1)
                inferred_code = infer_1["beams"][0]["inferred_code"] if infer_1["beams"] else ""
                schema_id = data.examples[index].schema.db_id
                fw.write(inferred_code + "\t" + schema_id + "\n")
