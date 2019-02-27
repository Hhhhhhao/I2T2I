from trainer.trainer_cls import Trainer
from torch.utils.data import DataLoader
from data_loader.txt2image_dataset import Text2ImageDataset_Origin
import argparse
from PIL import Image
import os

from utils.arg_extractor import get_args

if __name__ == '__main__':
    # args = get_args()

    train_data_loader = DataLoader(

        Text2ImageDataset_Origin(
            data_dir='/home/s1784380/I2T2I/data/',
            dataset_name="birds",
            which_set="train"
        ),

        batch_size=64,
        shuffle=False,
        num_workers=0
    )

    trainer = Trainer(gan_type="gan_cls",
                      train_data_loader=train_data_loader,
                      valid_data_loader=train_data_loader,
                      num_epochs=0,
                      lr=0,
                      save_path="/home/s1784380/lala/I2T2I",
                      l1_coef=0,
                      l2_coef=0,
                      pre_trained_disc="/home/s1784380/lala/I2T2I/saved/gan_cls/l1nl2/birds/checkpoints/disc_200.pth",
                      pre_trained_gen="/home/s1784380/lala/I2T2I/saved/gan_cls/l1nl2/birds/checkpoints/gen_200.pth",
                      )

    trainer.predict(data_loader=train_data_loader,valid=True,epoch=200)
