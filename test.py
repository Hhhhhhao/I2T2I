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
from torchvision import transforms
main_dir = os.path.dirname(__file__)
example_dir = os.path.join(main_dir, 'examples')


def main(config, resume):
    # setup data_loader instances
    # batch size must be one
    if "CoCo" in config["name"]:
        which_set = 'val'
        data_loader = getattr(module_data, config['train_data_loader']['type'])(
            # "/Users/leon/Projects/I2T2I/data/coco/",
            config['train_data_loader']['args']['data_dir'],
            which_set=which_set,
            image_size=256,
            batch_size=1,
            num_workers=0,
            validation_split=0
        )
    else:
        which_set = 'test'
        data_loader = getattr(module_data, config['train_data_loader']['type'])(
            # "/Users/leon/Projects/I2T2I/data/",
            config['train_data_loader']['args']['data_dir'],
            config['train_data_loader']['args']['dataset_name'],
            which_set=which_set,
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
    if torch.cuda.is_available():
        checkpoint = torch.load(resume)
    else:
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

    save_dir = os.path.dirname(resume)
    save_dir = os.path.dirname(resume)
    example_dir = os.path.join(save_dir, 'examples')
    if not os.path.exists(example_dir):
        os.mkdir(example_dir)

    gts = {}
    res = {}

    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)

    transform = transforms.Compose([
            transforms.Normalize(mean=(-mean/std).tolist(), std=(1.0/std).tolist()),
            transforms.ToPILImage()]
            )

    with torch.no_grad():
        for i, (batch_image_ids, batch_images, batch_captions, batch_caption_lengths) in enumerate(tqdm(data_loader)):
            batch_images, batch_captions = batch_images.to(device), batch_captions.to(device)
            # batch_caption_lengths = [l - 1 for l in batch_caption_lengths]
            if "Flowers" in config["name"] or "Birds" in config["name"]:
                img_id = batch_image_ids[0][:-2]
            elif "CoCo" in config["name"]:
                img_id = batch_image_ids[0]

            batch_features = model.encoder(batch_images)
            pred_captions = model.decoder.sample_beam_search(batch_features)

            pred_sentence = convert_back_to_text(list(pred_captions[0]), data_loader.dataset.vocab)
            target_sentence = convert_back_to_text(batch_captions.cpu().tolist()[0], data_loader.dataset.vocab)

            if i % 500 == 0:
                image = batch_images[0]
                image = transform(image.cpu())
                image.save(os.path.join(example_dir, '{}_{}.png'.format(img_id, pred_sentence)))

            # save sample images, or do something with output here
            if img_id not in gts.keys() and img_id not in res.keys():
                gts["{}".format(img_id)] = []
                res["{}".format(img_id)] = []

            if pred_sentence not in res["{}".format(img_id)]:
                res["{}".format(img_id)].append(pred_sentence)

            if target_sentence not in gts["{}".format(img_id)]:
                gts["{}".format(img_id)].append(target_sentence)

            # computing loss, metrics (for sentence) on test set
            # loss = loss_fn(targets, outputs)
            # batch_size = batch_images.shape[0]
            # total_loss += loss.item() * batch_size
            # for i, metric in enumerate(metric_fns):
            #    total_metrics[i] += metric(pred_sentence, target_sentence) * batch_size

    with open(os.path.join(save_dir, 'gts.json'), 'w') as f:
        json.dump(gts, f)

    with open(os.path.join(save_dir, 'res.json'), 'w') as f:
        json.dump(res, f)

    print("test images: {}".format(len(gts.keys())))

    # n_samples = len(data_loader.sampler)
    # log = {'loss': total_loss / n_samples}
    # log.update({met.__name__ : total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)})
    # print(log)

    return gts, res, save_dir


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')

    parser.add_argument('-r', '--resume', default='/Users/leon/Projects/I2T2I/saved/Show-and-Tell-Flowers/0212_233922/checkpoint-epoch5.pth', type=str,
                           help='path to latest checkpoint (default: None)')
    parser.add_argument('-d', '--device', default=None, type=str,
                           help='indices of GPUs to enable (default: all)')

    args = parser.parse_args()

    if args.resume:
        if torch.cuda.is_available():
            config = torch.load(args.resume)['config']
        else:
            config = torch.load(args.resume, map_location='cpu')['config']
    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    _, _, save_dir = main(config, args.resume)

    with open(os.path.join(save_dir, 'gts.json'), 'r') as f:
        gts = json.load(f)

    with open(os.path.join(save_dir, 'res.json'), 'r') as f:
        res = json.load(f)

    results = compute_score(gts, res)

    with open(os.path.join(save_dir, 'metrics.json'), 'w') as f:
        json.dump(results, f)

