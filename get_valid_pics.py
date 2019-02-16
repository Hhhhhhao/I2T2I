
from PIL import Image
from torch.utils.data import DataLoader
from data_loader.txt2image_dataset import Text2ImageDataset_Origin




if __name__ == '__main__':

    valid_data_loader = DataLoader(

        Text2ImageDataset_Origin(
            data_dir='/home/s1784380/I2T2I/data/',
            dataset_name="flowers",
            which_set="valid"
        ),

        batch_size=64,
        shuffle=False,
        num_workers=0
    )

    for batch_idx, sample in enumerate(valid_data_loader):
        if batch_idx != 0:
            break
        right_images = sample['right_images']
        txt = sample['txt']

        for image, t in zip(right_images, txt):
            im = Image.fromarray(image.data.mul_(127.5).add_(127.5).byte().permute(1, 2, 0).cpu().numpy())
            im.save("/home/s1784380/lala/I2T2I/valid_pic/flowers/ß{}.jpg".format(t.replace("/", "")[:100]))
