import os
import argparse
import torch
import json
import cv2
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from train import get_instance
from utils.util import convert_back_to_text
from eval_metrics.eval import compute_score
from torch.nn.utils.rnn import pack_padded_sequence


def main(config, resume):
    # setup data_loader instances
    # batch size must be one
    if "CoCo" in config["name"]:
        which_set = 'val'
        data_loader = getattr(module_data, config['train_data_loader']['type'])(
            "/Users/leon/Projects/I2T2I/data/coco/",
            # config['train_data_loader']['args']['data_dir'],
            which_set=which_set,
            image_size=128,
            batch_size=1,
            num_workers=0,
            validation_split=0
        )
    else:
        which_set = 'test'
        data_loader = getattr(module_data, config['train_data_loader']['type'])(
            "/Users/leon/Projects/I2T2I/data/",
            # config['train_data_loader']['args']['data_dir'],
            config['train_data_loader']['args']['dataset_name'],
            which_set=which_set,
            image_size=128,
            batch_size=1,
            num_workers=0
        )

    # build model architecture
    model = get_instance(module_arch, 'arch', config)
    model.summary()

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss'])
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    # load state dict
    checkpoint = torch.load(resume, map_location='cpu')
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))

    gts = {}
    res = {}

    with torch.no_grad():
        for i, (batch_image_ids, batch_images, batch_captions, batch_caption_lengths) in enumerate(tqdm(data_loader)):
            batch_images, batch_captions = batch_images.to(device), batch_captions.to(device)
            # batch_caption_lengths = [l - 1 for l in batch_caption_lengths]

            batch_features = model.encoder(batch_images)
            pred_captions= model.decoder.sample_beam_search(batch_features, max_len=20, beam_width=3)

            pred_sentence = convert_back_to_text(pred_captions[0], data_loader.dataset.vocab)
            # pred_sentence_2 = convert_back_to_text(pred_captions_greedy, data_loader.dataset.vocab)
            target_sentence = convert_back_to_text((batch_captions.numpy())[0], data_loader.dataset.vocab)

            print("prediction: {}".format(pred_sentence))
            print("ground truth: {}".format(target_sentence))
            # save sample images, or do something with output here
            gts["{}".format(batch_image_ids[0])] = target_sentence
            res["{}".format(batch_image_ids[0])] = pred_sentence

            # computing loss, metrics (for sentence) on test set
            # loss = loss_fn(targets, outputs)
            # batch_size = batch_images.shape[0]
            # total_loss += loss.item() * batch_size
            # for i, metric in enumerate(metric_fns):
            #    total_metrics[i] += metric(pred_sentence, target_sentence) * batch_size

    with open('saved/{}/gts.json'.format(config["name"]), 'w') as f:
        json.dump(gts, f)

    with open('saved/{}/res.json'.format(config["name"]), 'w') as f:
        json.dump(res, f)

    results = compute_score(gts, res)

    with open('saved/{}/metrics.json'.format(config["name"]), 'w') as f:
        json.dump(results, f)

    n_samples = len(data_loader.sampler)
    log = {'loss': total_loss / n_samples}
    log.update({met.__name__ : total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)})
    print(log)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')

    parser.add_argument('-r', '--resume', default='/Users/leon/Projects/I2T2I/saved/Show-and-Tell-Birds/0210_195346/model_best.pth', type=str,
                           help='path to latest checkpoint (default: None)')
    parser.add_argument('-d', '--device', default=None, type=str,
                           help='indices of GPUs to enable (default: all)')

    args = parser.parse_args()

    if args.resume:
        config = torch.load(args.resume, map_location='cpu')['config']
    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"]=args.device
    main(config, args.resume)
