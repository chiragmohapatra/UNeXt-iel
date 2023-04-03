import argparse
import os
from glob import glob

import cv2
import torch
import torch.backends.cudnn as cudnn
import yaml
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import archs
from dataset import Dataset
from metrics import iou_score
from utils import AverageMeter
from albumentations import RandomRotate90,Resize
import time
from archs import UNext

arch = 'UNext'
input_channels = 3
deep_supervision = False


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_classes', default=6, help='number of classes')
    parser.add_argument('--src', default=None,required=True, help='directory of the source images')
    parser.add_argument('--img_ext', default='.png', help='image extension')
    parser.add_argument('--weights', default=None,required=True, help='model weights')
    parser.add_argument('--img_sz', default=640, help='size of input images')
    parser.add_argument('--batch_size', default=1, help='batch size')
    parser.add_argument('--num_workers', default=4, help='number of workers')
    parser.add_argument('--save_dir', default='outputs', help='directory to save outputs')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    cudnn.benchmark = True

    print("=> creating model UNext")
    model = archs.__dict__[arch](args.num_classes,
                                input_channels,
                                deep_supervision)

    model = model.cuda()

    # Data loading code
    test_img_ids = glob(os.path.join(args.src, '*' + args.img_ext))
    test_img_ids = [os.path.splitext(os.path.basename(p))[0] for p in test_img_ids]

    model.load_state_dict(torch.load(args.weights))
    model.eval()

    val_transform = Compose([
        Resize(args.img_sz, args.img_sz),
        transforms.Normalize(),
    ])

    val_dataset = Dataset(
        img_ids=test_img_ids,
        img_dir=args.src,
        img_ext=args.img_ext,
        num_classes=args.num_classes,
        transform=val_transform)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        drop_last=False)

    for c in range(args.num_classes):
        os.makedirs(os.path.join('outputs', str(c)), exist_ok=True)
    with torch.no_grad():
        for input, meta in tqdm(val_loader, total=len(val_loader)):
            input = input.cuda()
            model = model.cuda()
            # compute output
            output = model(input)

            output = torch.sigmoid(output).cpu().numpy()
            output[output>=0.5]=1
            output[output<0.5]=0

            for i in range(len(output)):
                for c in range(args.num_classes):
                    cv2.imwrite(os.path.join('outputs', str(c), meta['img_id'][i] + '.png'),
                                (output[i, c] * 255).astype('uint8'))

    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
