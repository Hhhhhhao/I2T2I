import os
import json
import argparse
import torch
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from trainer import DAMSM_Trainer
from utils import Logger


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

    # build model architecture
    model = get_instance(module_arch, 'arch', config)
    print(model)

    # get function handles of loss and metrics
    losses = {}
    losses["word"] = getattr(module_loss, 'words_loss')
    losses["sent"] = getattr(module_loss, 'sent_loss')
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = get_instance(torch.optim, 'optimizer', config, trainable_params)
    lr_scheduler = None # get_instance(torch.optim.lr_scheduler, 'lr_scheduler', config, optimizer)

    trainer = DAMSM_Trainer(model, losses, metrics, optimizer,
                      resume=resume,
                      config=config,
                      data_loader=train_data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler,
                      train_logger=train_logger)

    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('-c', '--config', default='config/birds_damsm_config.json', type=str,
                        help='config file path (default: None)')
    parser.add_argument('-r', '--resume', default=None, type=str,
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
        raise AssertionError("Configuration file need to be specified. Add '-c config.json', for example.")

    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    main(config, args.resume)