import numpy as np
import torch
import dataloaders.trans as trans
from dataloaders.dataloader import MyDataloader
# import trans
# from dataloader import MyDataloader
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import os
import cv2
import matplotlib.pyplot as plt

iheight, iwidth = 480, 640 # raw image size

    # depth = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

class EgoGesture(MyDataloader):
    def __init__(self,
                 root,
                 split,
                 annotation_path, 
                 hdf5=False, 
                 modality='rgb',
                 min_depth=1e-3,
                 max_depth=np.inf):
        self.split = split
        super(EgoGesture, self).__init__(root, split, annotation_path, hdf5, modality)
        self.output_size = (224, 224)
        self.min_depth = min_depth
        self.max_depth = max_depth

    # def is_image_file(self, filename):
        # # IMG_EXTENSIONS = ['.h5']
        # if self.split == 'train':
            # return (filename.endswith('.h5') and \
                # '00001.h5' not in filename and '00201.h5' not in filename)
        # elif self.split == 'holdout':
            # return ('00001.h5' in filename or '00201.h5' in filename)
        # elif self.split == 'val':
            # return (filename.endswith('.h5'))
        # else:
            # raise (RuntimeError("Invalid dataset split: " + self.split + "\n"
                                # "Supported dataset splits are: train, val"))

    def train_transform(self, rgb, depth):
        """
        Input is ndarray with shape (H, W, C)
        Output is an ndarray with shape (H, W, C)
        """
        # rgb_np = rgb
        # depth_np = depth
        # depth_gt_np = depth
        s = np.random.uniform(1.0, 1.5) # random scaling
        angle = np.random.uniform(-5.0, 5.0) # random rotation degrees
        do_flip = np.random.uniform(0.0, 1.0) < 0.5 # random horizontal flip

        transform = transforms.Compose([
            transforms.ToPILImage(), 
            transforms.RandomRotation((angle, angle), interpolation=InterpolationMode.BILINEAR), 
            # transforms.Resize((int(np.ceil(360/s)), int(np.ceil(480/s))), InterpolationMode.BILINEAR), 
            # transforms.Resize((270, 360), interpolation=InterpolationMode.BILINEAR), 
            transforms.RandomHorizontalFlip(do_flip), 
            transforms.CenterCrop((224, 224)), 
            ])
        # transform_depth = transforms.Compose([
            # transforms.ToPILImage(), 
            # transforms.RandomRotation((angle, angle), interpolation=InterpolationMode.BILINEAR), 
            # # transforms.Resize((int(np.ceil(360/s)), int(np.ceil(480/s))), InterpolationMode.BILINEAR), 
            # transforms.Resize((270, 360), interpolation=InterpolationMode.BILINEAR), 
            # transforms.RandomHorizontalFlip(do_flip), 
            # transforms.CenterCrop((224, 224)), 
            # ])
        medianpool = trans.MedianPooling(p=8, sigma=0, min_depth=self.min_depth, max_depth=self.max_depth)
        sparseSample = trans.RandomSparseSampling(n=500)
        # color_jitter = transforms.ColorJitter(0.4, 0.4, 0.4)

        rgb = transform(rgb)
        # rgb = color_jitter(rgb) # random color jittering
        depth = transform(depth)
        rgb_np = np.asfarray(rgb, dtype='float') / 255
        # print(rgb_np)
        # rgb_np = np.array(rgb)
        depth_np = np.array(depth)  # shape H, W, C
        depth_gt_np = np.array(depth)
        depth_np = np.reshape(depth_np, (depth_np.shape[0], depth_np.shape[1], 1))
        depth_gt_np = np.reshape(depth_gt_np, (depth_np.shape[0], depth_np.shape[1], 1))

        depth_np = medianpool(depth_np)

        # Scale 0-1
        depth_np = depth_np/1000
        # depth_gt_np = depth_gt_np/1000
        # depth_np = sparseSample(depth_np)
        # plt.subplot(131)
        # plt.imshow(rgb_np)
        # plt.subplot(132)
        # plt.imshow(depth_gt_np)
        # plt.subplot(133)
        # plt.imshow(depth_np)
        # plt.show()

        return rgb_np, depth_np, depth_gt_np

    def val_transform(self, rgb, depth):
        """
        Input is an ndarray with shape (H, W, C)
        Output is an ndarray with shape (H, W, C)
        """
        # rgb_np = rgb
        # depth_np = depth
        # depth_gt_np = depth
        transform = transforms.Compose([
            transforms.ToPILImage(), 
            # transforms.Resize((int(np.ceil(360/s)), int(np.ceil(480/s))), InterpolationMode.BILINEAR), 
            # transforms.Resize((270, 360), interpolation=InterpolationMode.BILINEAR), 
            transforms.CenterCrop((224, 224)), 
            ])
        # transform_depth = transforms.Compose([
            # transforms.ToPILImage(), 
            # # transforms.Resize((int(np.ceil(360/s)), int(np.ceil(480/s))), InterpolationMode.BILINEAR), 
            # transforms.Resize((270, 360), interpolation=InterpolationMode.BILINEAR), 
            # transforms.CenterCrop((224, 224)), 
            # ])
        # # transform = transforms.Compose([
            # # transforms.ToPILImage(), 
            # # transforms.Resize((264, 352)), 
            # # transforms.CenterCrop((224, 224)), 
            # # ])
        sparseSample = trans.RandomSparseSampling(n=500)
        medianpool = trans.MedianPooling(p=8, sigma=0, min_depth=self.min_depth, max_depth=self.max_depth)

        rgb = transform(rgb)
        depth = transform(depth)
        rgb_np = np.asfarray(rgb, dtype='float') / 255
        depth_np = np.array(depth)  # shape H, W, C
        depth_gt_np = np.array(depth)
        depth_np = np.reshape(depth_np, (depth_np.shape[0], depth_np.shape[1], 1))
        depth_gt_np = np.reshape(depth_gt_np, (depth_np.shape[0], depth_np.shape[1], 1))

        depth_np = medianpool(depth_np)
        # Scale 0-1
        depth_np = depth_np/1000
        # depth_gt_np = depth_gt_np/1000
        # depth_np = sparseSample(depth_np)

        return rgb_np, depth_np, depth_gt_np
