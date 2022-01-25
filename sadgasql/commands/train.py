import datetime
import json
import os
import _jsonnet
import attr
import torch
from sadgasql.utils import random_state
from sadgasql.utils import registry
from sadgasql.utils import saver as saver_mod


@attr.s
class TrainConfig:
    report_every_n = attr.ib(default=100)
    save_every_n = attr.ib(default=100)
    keep_every_n = attr.ib(default=1000)
    batch_size = attr.ib(default=20)
    max_steps = attr.ib(default=40000)
    data_seed = attr.ib(default=None)
    init_seed = attr.ib(default=None)
    model_seed = attr.ib(default=None)
    num_batch_accumulated = attr.ib(default=1)
    clip_grad = attr.ib(default=None)


class Logger:
    def __init__(self, log_path=None, reopen_to_flush=False):
        self.log_file = None
        self.reopen_to_flush = reopen_to_flush
        if log_path is not None:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            self.log_file = open(log_path, 'a+')

    def log(self, msg):
        formatted = f'[{datetime.datetime.now().replace(microsecond=0).isoformat()}] {msg}'
        print(formatted)
        if self.log_file:
            self.log_file.write(formatted + '\n')
            if self.reopen_to_flush:
                log_path = self.log_file.name
                self.log_file.close()
                self.log_file = open(log_path, 'a+')
            else:
                self.log_file.flush()

class Trainer:
    def __init__(self, logger, config):
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.logger = logger
        self.train_config = registry.instantiate(TrainConfig, config['train'])
        self.data_random = random_state.RandomContext(self.train_config.data_seed)
        self.model_random = random_state.RandomContext(self.train_config.model_seed)
        self.init_random = random_state.RandomContext(self.train_config.init_seed)

        with self.init_random:
            self.model_preproc = registry.instantiate(
                registry.lookup('model', config['model']).Preproc,
                config['model'],
                unused_keys=('name',))
            self.model_preproc.load()
            self.model = registry.construct('model', config['model'],
                                            unused_keys=('encoder_preproc', 'decoder_preproc'),
                                            preproc=self.model_preproc, device=self.device)
            self.model.to(self.device)

    def train(self, config, args):
        modeldir = args.logdir
        with self.init_random:
            if config["optimizer"].get("name", None) == 'bertAdamw':
                bert_params = list(self.model.encoder.bert_model.parameters())
                assert len(bert_params) > 0
                non_bert_params = []
                for name, _param in self.model.named_parameters():
                    if "bert" not in name:
                        non_bert_params.append(_param)
                assert len(non_bert_params) + len(bert_params) == len(list(self.model.parameters()))

                optimizer = registry.construct('optimizer', config['optimizer'], non_bert_params=non_bert_params,
                                               bert_params=bert_params)
                lr_scheduler = registry.construct('lr_scheduler',
                                                  config.get('lr_scheduler', {'name': 'noop'}),
                                                  param_groups=[optimizer.non_bert_param_group,
                                                                optimizer.bert_param_group])
            else:
                optimizer = registry.construct('optimizer', config['optimizer'], params=self.model.parameters())
                lr_scheduler = registry.construct('lr_scheduler',
                                                      config.get('lr_scheduler', {'name': 'noop'}),
                                                      param_groups=optimizer.param_groups)
        saver = saver_mod.Saver(
            {"model": self.model, "optimizer": optimizer}, keep_every_n=self.train_config.keep_every_n)
        last_step = saver.restore(modeldir, map_location=self.device)

        if "pretrain" in config and last_step == 0:
            pretrain_config = config["pretrain"]
            _path = pretrain_config["pretrained_path"]
            _step = pretrain_config["checkpoint_step"]
            pretrain_step = saver.restore(_path, step=_step, map_location=self.device, item_keys=["model"])
            saver.save(modeldir, pretrain_step)
            last_step = pretrain_step

        with self.data_random:
            train_data = self.model_preproc.dataset('train')
            train_data_loader = self._yield_batches_from_epochs(
                torch.utils.data.DataLoader(
                    train_data,
                    batch_size=self.train_config.batch_size,
                    shuffle=True,
                    drop_last=True,
                    collate_fn=lambda x: x))

        with self.data_random:
            for batch in train_data_loader:
                if last_step >= self.train_config.max_steps:
                    break

                with self.model_random:
                    for _i in range(self.train_config.num_batch_accumulated):
                        if _i > 0:
                            batch = next(train_data_loader)
                        loss = self.model.compute_loss(batch)
                        norm_loss = loss / self.train_config.num_batch_accumulated
                        norm_loss.backward()

                    if self.train_config.clip_grad:
                        torch.nn.utils.clip_grad_norm_(optimizer.bert_param_group["params"], \
                                                       self.train_config.clip_grad)
                    optimizer.step()
                    lr_scheduler.update_lr(last_step)
                    optimizer.zero_grad()

                if last_step % self.train_config.report_every_n == 0:
                    self.logger.log(f'Step {last_step}: loss={loss.item():.4f}')

                last_step += 1
                if last_step == 1 or last_step % self.train_config.save_every_n == 0:
                    saver.save(modeldir, last_step)
            saver.save(modeldir, last_step)

    @staticmethod
    def _yield_batches_from_epochs(loader):
        while True:
            for batch in loader:
                yield batch

def main(args):
    config = json.loads(_jsonnet.evaluate_file(args.config))
    logger = Logger(os.path.join(args.logdir, 'log.txt'), reopen_to_flush=config.get('log', {}).get('reopen_to_flush'))

    with open(os.path.join(args.logdir,
                           f'config-{datetime.datetime.now().strftime("%Y%m%dT%H%M%S%Z")}.json'), 'w') as f:
        json.dump(config, f, sort_keys=True, indent=4)

    logger.log(f'Logging to {args.logdir}')

    trainer = Trainer(logger, config)
    trainer.train(config, args)