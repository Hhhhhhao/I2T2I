import os
import argparse
import torch
import json
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from train import get_instance
from utils.util import convert_back_to_text
from eval_metrics.eval import compute_score


def main(config, resume):
    # setup data_loader instances
    # batch size must be one
    data_loader = getattr(module_data, config['train_data_loader']['type'])(
        '/Users/leon/Projects/I2T2I/data/',# config['train_data_loader']['args']['data_dir'],
        config['train_data_loader']['args']['dataset_name'],
        which_set='test',
        image_size=256,
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

            batch_features = model.encoder(batch_images)
            pred_captions = model.decoder(batch_features)
            pred_captions_cpu = pred_captions.cpu().data.numpy()
            pred_sentence = convert_back_to_text(pred_captions_cpu, data_loader.dataset.vocab)
            target_sentence = convert_back_to_text(batch_captions, data_loader.dataset.vocab)
            print(pred_sentence)
            print(target_sentence)

            # save sample images, or do something with output here
            gts["{}".format(batch_image_ids[0])] = target_sentence
            res["{}".format(batch_image_ids[0])] = pred_sentence

            # computing loss, metrics (for sentence) on test set
            loss = loss_fn(pred_captions, batch_captions)
            batch_size = batch_images.shape[0]
            total_loss += loss.item() * batch_size
            for i, metric in enumerate(metric_fns):
                total_metrics[i] += metric(pred_sentence, target_sentence) * batch_size

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

    parser.add_argument('-r', '--resume', default='/Users/leon/Projects/I2T2I/saved/Show-and-Tell-Flowers/0205_233748/model_best.pth', type=str,
                           help='path to latest checkpoint (default: None)')
    parser.add_argument('-d', '--device', default=None, type=str,
                           help='indices of GPUs to enable (default: all)')

    args = parser.parse_args()

    if args.resume:
        config = torch.load(args.resume, map_location='cpu')['config']
    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"]=args.device
    main(config, args.resume)
