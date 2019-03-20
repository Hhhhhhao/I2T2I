import os
import json
import argparse
import torch
import data_loader.data_loaders as module_data
import model.model as module_arch
from trainer.gan_trainer import Trainer
from utils import Logger
dirname = os.path.dirname(__file__)
BIRD_PATH = os.path.join(dirname, 'saved/Show-and-Tell-Birds/0218_112224/model_best.pth')
FLOWER_PATH = os.path.join(dirname, 'saved/Show-and-Tell-Flowers/0218_112128/model_best.pth')


def get_instance(module, name, config, *args):
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])


def main(config, resume):
    train_logger = Logger()

    # setup data_loader instances
    train_data_loader = get_instance(module_data, 'train_data_loader', config)
    print("train vocab size:{}".format(len(train_data_loader.dataset.vocab)))

    if "CoCo" in config['name']:
        valid_data_loader = train_data_loader.split_validation()
    else:
        valid_data_loader = get_instance(module_data, 'valid_data_loader', config)
        print("val vocab size:{}".format(len(valid_data_loader.dataset.vocab)))

    # get function handles of loss and metrics
    loss = torch.nn.MSELoss()
    metrics = None

    model_config = config["models"]
    models = {}
    optimizers = {}
    # initialize generator and discriminator from config
    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    models["generator"] = get_instance(module_arch, 'Generator', model_config)
    generator_trainable_params = filter(lambda p: p.requires_grad, models["generator"].parameters())
    models["discriminator"] = get_instance(module_arch, 'Discriminator', model_config)
    discriminator_trainable_params = filter(lambda p: p.requires_grad, models["discriminator"] .parameters())
    optimizers["generator"] = get_instance(torch.optim, 'optimizer', model_config["Generator"],
                                            generator_trainable_params)
    optimizers["discriminator"] = get_instance(torch.optim, 'optimizer', model_config["Discriminator"],
                                                discriminator_trainable_params)

    trainer = Trainer(models=models,
                      optimizers=optimizers,
                      losses=loss,
                      metrics=metrics,
                      resume=resume,
                      config=config,
                      data_loader=train_data_loader,
                      valid_data_loader=valid_data_loader,
                      train_logger=train_logger)
    # trainer.pre_train()
    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('-c', '--config', default="config/flowers_config.json", type=str,
                        help='config file path (default: None)')
    parser.add_argument('-r', '--resume', default='/Users/leon/Downloads/hdgan/HDGAN-Birds/0315_222437/checkpoint-epoch73.pth', type=str,
                        help='path to latest checkpoint (default: None)')
    parser.add_argument('-d', '--device', default=None, type=str,
                        help='indices of GPUs to enable (default: all)')
    args = parser.parse_args()

    if args.config:
        # load config file
        config = json.load(open(args.config))
        path = os.path.join(config['trainer']['save_dir'], config['name'])
    elif args.resume:
        # load config file from checkpoint, in case new config file is not given.
        # Use '--config' and '--resume' arguments together to load trained model and train more with changed config.
        config = torch.load(args.resume)['config']
    else:
        raise AssertionError("Configuration file need to be specified. Add '-c birds_config.json', for example.")

    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    main(config, args.resume)