from trainer.trainer_cls import Trainer
from data_loader.data_loaders import Text2ImageDataLoader
import argparse
from PIL import Image
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gan_type", default='gan_cls')
    parser.add_argument('--num_epochs', default=200, type=int)
    parser.add_argument("--lr", default=0.0002, type=float)
    parser.add_argument("--l1_coef", default=50, type=float)
    parser.add_argument("--l2_coef", default=100, type=float)

    # TODO: 'env_name'???
    parser.add_argument("--vis_screen", default='I2T2I')

    parser.add_argument("--save_path", default='/home/s1784380/lala/I2T2I/saved/GAN-CLS')
    parser.add_argument('--pre_trained_disc', default=None)
    parser.add_argument('--pre_trained_gen', default=None)

    args = parser.parse_args()

    bird_train_data_loader = Text2ImageDataLoader(
        data_dir='/home/s1784380/I2T2I/data/',
        dataset_name="flowers",
        which_set="train",
        image_size=64,
        batch_size=16,
        num_workers=0
    )

    trainer = Trainer(gan_type=args.gan_type,
                      data_loader=bird_train_data_loader,
                      num_epochs=args.num_epochs,
                      lr=args.lr,
                      save_path=args.save_path,
                      l1_coef=args.l1_coef,
                      l2_coef=args.l2_coef,
                      pre_trained_disc=args.pre_trained_disc,
                      pre_trained_gen=args.pre_trained_gen,
                      )

    trainer.train()