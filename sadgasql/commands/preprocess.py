import json

import _jsonnet
import tqdm

# noinspection PyUnresolvedReferences
from sadgasql import datasets
# noinspection PyUnresolvedReferences
from sadgasql import grammars
# noinspection PyUnresolvedReferences
from sadgasql import models
# noinspection PyUnresolvedReferences
from sadgasql.utils import registry
# noinspection PyUnresolvedReferences
from sadgasql.utils import vocab
from sadgasql.datasets import spider


class Preprocessor:
    def __init__(self, config):
        self.config = config
        self.model_preproc = registry.instantiate(
            registry.lookup('model', config['model']).Preproc,
            config['model'],
            unused_keys=('name',))

    def preprocess(self):
        self.model_preproc.clear_items()
        for section in self.config['data']:
            data = spider.SpiderDataset(paths=self.config['data'][section]['paths'], tables_paths=self.config['data'][section]['tables_paths'], db_path=self.config['data'][section]['db_path'])
            # data = registry.construct('dataset', self.config['data'][section])
            for item in tqdm.tqdm(data, desc=f"{section} section", dynamic_ncols=True):
                to_add, validation_info = self.model_preproc.validate_item(item, section)
                if to_add:
                    self.model_preproc.add_item(item, section, validation_info)
        self.model_preproc.save()
def main(args):
    config = json.loads(_jsonnet.evaluate_file(args.config))
    preprocessor = Preprocessor(config)
    preprocessor.preprocess()