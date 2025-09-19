#!/usr/bin/env python

# wujian@2018

import os
import pprint
import argparse
import random

from libs.trainer import SiSnrTrainer
from libs.dataset import make_dataloader
from libs.utils import dump_json, get_logger
from model import Model

from conf import trainer_conf, train_data, dev_data, eval_data, chunk_size,checkpoint,epochs,batch_size,unet_conf,start_end

logger = get_logger(__name__)


def run(args):
    gpuids = tuple(map(int, args.gpus.split(",")))


    nnet = Model()
    # nnet = ConvTasNet(**nnet_conf)
    trainer = SiSnrTrainer(nnet,
                           gpuid=gpuids,
                           checkpoint=checkpoint,
                           resume=args.resume,
                           eval_path = args.eval_path,
                           **trainer_conf,
                           start_end=start_end)

    data_conf = {
        "train": train_data,
        "dev": dev_data,
        "eval": eval_data,
        "chunk_size": chunk_size
    }
    for conf, fname in zip([unet_conf, trainer_conf, data_conf],
                           ["mdl.json", "trainer.json", "data.json"]):
        dump_json(conf, checkpoint, fname)

    train_loader = make_dataloader(train=True,
                                   data_kwargs=train_data,
                                   batch_size=batch_size,
                                   chunk_size=chunk_size,
                                   num_workers=args.num_workers)
    dev_loader = make_dataloader(train=False,
                                 data_kwargs=dev_data,
                                 batch_size=batch_size,
                                 chunk_size=chunk_size,
                                 num_workers=args.num_workers)

    eval_loader = make_dataloader(train=False,
                                 data_kwargs=eval_data,
                                 batch_size=batch_size,
                                 chunk_size=chunk_size,
                                 num_workers=args.num_workers)

    trainer.run(train_loader, dev_loader,eval_loader, num_epochs=epochs)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=
        "Command to start ConvTasNet training, configured from conf.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--gpus",
                        type=str,
                        default="0,1,2,3",
                        help="Training on which GPUs "
                        "(one or more, egs: 0, \"0,1\")")

    parser.add_argument("--resume",
                        type=str,
                        default="",
                        help="Exist model to resume training from")
    parser.add_argument("--eval_path",
                        type=str,
                        default=None,
                        help="just eval best model")
    parser.add_argument("--num-workers",
                        type=int,
                        default=1,
                        help="Number of workers used in data loader")
    args = parser.parse_args()
    logger.info("Arguments in command:\n{}".format(pprint.pformat(vars(args))))

    run(args)
