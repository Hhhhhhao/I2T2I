from trainer.trainer_cls import Trainer
from data_loader.data_loaders import Text2ImageDataLoader
import argparse
from PIL import Image
import os

from utils.arg_extractor import get_args

if __name__ == '__main__':

    args = get_args()

    if args.dataset_name == 'birds':

        train_data_loader = Text2ImageDataLoader(
            data_dir='/home/s1784380/I2T2I/data/',
            dataset_name="birds",
            which_set="train",
            image_size=64,
            batch_size=16,
            num_workers=0
        )
    elif args.dataset_name == 'flowers':

        train_data_loader = Text2ImageDataLoader(
            data_dir='/home/s1784380/I2T2I/data/',
            dataset_name="flowers",
            which_set="train",
            image_size=64,
            batch_size=16,
            num_workers=0
        )
    else:
        raise AssertionError("dataset_name not valid!")

    trainer = Trainer(gan_type=args.gan_type,
                      data_loader=train_data_loader,
                      num_epochs=args.num_epochs,
                      lr=args.lr,
                      save_path=args.save_path,
                      l1_coef=args.l1_coef,
                      l2_coef=args.l2_coef,
                      pre_trained_disc=args.pre_trained_disc,
                      pre_trained_gen=args.pre_trained_gen,
                      )

    trainer.train()
