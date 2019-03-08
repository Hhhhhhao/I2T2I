import argparse
import json



def get_args():
    """
    Returns a namedtuple with arguments extracted from the command line.
    :return: A namedtuple with arguments
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--gan_type", default='lsgan_mbd_cls', help="GAN type (str): 'gan_cls' ")
    parser.add_argument("--dataset_name", default="birds", help="dataset name (str): default 'birds', 'flowers' ")
    parser.add_argument('--num_epochs', default=300, type=int, help="num of ephochs for training (int): default 200 ")
    parser.add_argument("--lr", default=0.0002, type=float, help="learning rate (float): 0.0002")
    parser.add_argument("--l1_coef", default=50, type=float, help="l1 coefficient for discriminator loss")
    parser.add_argument("--l2_coef", default=100, type=float, help="l2 coefficient for discriminator loss")
    parser.add_argument("--save_path", default=None, help="checkpoints save path")
    parser.add_argument('--pre_trained_disc', default=None, help="Generator pre-tranined model path used for intializing training")
    parser.add_argument('--pre_trained_gen', default=None, help="Discriminator pre-tranined model path used for intializing training")

    parser.add_argument('--filepath_to_config_file', default='config/birds_config_wgan.json', help='file path to arguments json file')

    args = parser.parse_args()

    if args.filepath_to_config_file is not None:
        args = extract_args_from_json(json_file_path=args.filepath_to_config_file, existing_args_dict=args)

    arg_str = [(str(key), str(value)) for (key, value) in vars(args).items()]
    print(arg_str)
    return args


class AttributeAccessibleDict(object):
    def __init__(self, adict):
        self.__dict__.update(adict)


def extract_args_from_json(json_file_path, existing_args_dict=None):

    summary_filename = json_file_path
    with open(summary_filename) as f:
        arguments_dict = json.load(fp=f)

    for key, value in vars(existing_args_dict).items():
        if key not in arguments_dict:
            arguments_dict[key] = value

    arguments_dict = AttributeAccessibleDict(arguments_dict)

    return arguments_dict
