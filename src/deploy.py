# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     deploy
   Description :
   Author :       haxu
   date：          2019-05-29
-------------------------------------------------
   Change Activity:
                   2019-05-29:
-------------------------------------------------
"""
__author__ = 'haxu'

import os
import torch
from argparse import Namespace
from networks import get_network
import transforms as jnt_trnsf
import torchvision.transforms as std_trnsf
from data_loader import get_loader
import time
import cv2
import numpy as np

from matplotlib import pyplot as plt

args = Namespace(
    name='pspnet_resnet101',
    ckpt_dir='../checkpoints/pspnet_resnet101_sgd_lr_0.002_epoch_100_test_iou_0.918.pth',
    data_dir='../data/Figaro1k',
    dataset='figaro',
    save_dir='../experiment/'
)


def main():
    net = get_network(args.name)
    state = torch.load(args.ckpt_dir, map_location=torch.device('cpu'))
    net.load_state_dict(state['weight'])

    test_joint_transforms = jnt_trnsf.Compose([
        jnt_trnsf.Safe32Padding()
    ])

    test_image_transforms = std_trnsf.Compose([
        std_trnsf.ToTensor(),
        std_trnsf.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # transforms only on mask
    mask_transforms = std_trnsf.Compose([
        std_trnsf.ToTensor()
    ])

    test_loader = get_loader(dataset=args.dataset,
                             data_dir=args.data_dir,
                             train=False,
                             joint_transforms=test_joint_transforms,
                             image_transforms=test_image_transforms,
                             mask_transforms=mask_transforms,
                             batch_size=1,
                             shuffle=False,
                             num_workers=4)

    img_dir = os.path.join(args.data_dir, 'Original', 'Testing')
    imgs = [os.path.join(img_dir, k) for k in sorted(os.listdir(img_dir)) if k.endswith('.jpg')]

    durations = []

    with torch.no_grad():
        net.eval()
        for i, (data, label) in enumerate(test_loader):
            print('[{:3d}/{:3d}] processing image... '.format(i + 1, len(test_loader)))
            start = time.time()
            logit = net(data)
            duration = time.time() - start
            durations.append(duration)

            pred = torch.sigmoid(logit.cpu())[0][0].data.numpy()
            mh, mw = data.size(2), data.size(3)
            mask = pred >= 0.5

            mask_n = np.zeros((mh, mw, 3))
            mask_n[:, :, 2] = 255
            mask_n[:, :, 2] *= mask

            path = os.path.join(args.save_dir, "figaro_img_%04d.png" % i)
            image_n = cv2.imread(imgs[i])

            # discard padded area
            ih, iw, _ = image_n.shape

            delta_h = mh - ih
            delta_w = mw - iw

            top = delta_h // 2
            bottom = mh - (delta_h - top)
            left = delta_w // 2
            right = mw - (delta_w - left)

            mask_n = mask_n[top:bottom, left:right, :]

            image_n = image_n * 0.7 + mask_n * 0.3

            cv2.imwrite(path, image_n)

            break

    print(f'fps avg {len(durations) / sum(durations)} ')


if __name__ == '__main__':
    main()
