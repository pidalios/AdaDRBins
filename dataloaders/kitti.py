import numpy as np
from torchvision import transforms
from dataloaders.dataloader import MyDataloader
from torchvision.transforms.functional import InterpolationMode
import matplotlib.pyplot as plt
# from dataloaders.trans import UniformSampling
from dataloaders.trans import *
from scipy.interpolate import LinearNDInterpolator

def lin_interp(shape, xyd):
    # taken from https://github.com/hunse/kitti
    m, n = shape
    ij, d = xyd[:, 1::-1], xyd[:, 2]
    f = LinearNDInterpolator(ij, d, fill_value=0)
    J, I = np.meshgrid(np.arange(n), np.arange(m))
    IJ = np.vstack([I.flatten(), J.flatten()]).T
    disparity = f(IJ).reshape(shape)
    return disparity

class KITTIDataset(MyDataloader):
    # def __init__(self, root, split, sparsifier=None, modality='rgb'):
    def __init__(self,
                 root,
                 split,
                 annotation_path=None, 
                 hdf5=True, 
                 modality='rgb',
                 samples=10000, 
                 sparse_gt=False,
                 min_depth=1e-3,
                 max_depth=np.inf):
        # super(KITTIDataset, self).__init__(root, split, sparsifier, modality)
        super(KITTIDataset, self).__init__(root, split, annotation_path, hdf5, modality)
        self.samples = samples
        self.sparse_gt = sparse_gt

    def train_transform(self, rgb, depth):
        depth = np.asfarray(depth, dtype='float32')
        s = np.random.uniform(1.0, 1.5)  # random scaling
        # depth_np = depth / s
        angle = np.random.uniform(-2.0, 2.0)  # random rotation degrees
        do_flip = np.random.uniform(0.0, 1.0) < 0.5  # random horizontal flip

        # perform 1st step of data augmentation
        rgb = transforms.functional.to_pil_image(rgb, mode=None)
        depth = transforms.functional.to_pil_image(depth, mode=None)
        rgb = transforms.functional.crop(rgb, 140, 10, 224, 1216)
        depth = transforms.functional.crop(depth, 140, 10, 224, 1216)

        # rgb = transforms.functional.crop(rgb, 140, 256, 224, 704)
        # depth = transforms.functional.crop(depth, 140, 256, 224, 704)
        transform = transforms.Compose([
            # transforms.CenterCrop((352, 1216)), 
            transforms.RandomHorizontalFlip(do_flip), 
        ])
        sample = UniformSampling(num_samples=self.samples)
        # sample = Drop_depth(prob_keep=0.3)
        rgb_np = transform(rgb)
        depth_np = transform(depth)
        rgb_np = np.asfarray(rgb_np, dtype='float') / 255
        depth_np = np.asfarray(depth_np, dtype='float32')
        gt_np = depth_np


        # Dense depth interpolation
        if ~self.sparse_gt:
            x, y = np.where(gt_np > 0)
            d = gt_np[gt_np != 0]
            xyd = np.stack((y, x, d)).T
            gt_np = lin_interp(gt_np.shape, xyd)

        gt_np = np.reshape(gt_np, (gt_np.shape[0], gt_np.shape[1], 1))


        # print(np.count_nonzero(gt_np))
        depth_np = np.reshape(depth_np, (depth_np.shape[0], depth_np.shape[1], 1))
        depth_np = sample(depth_np)
        # depth_np = drop_depth_measurements(depth_np, 0.3)
        # print(np.count_nonzero(depth_np))
        # print('--')
        # plt.subplot(311)
        # plt.imshow(rgb_np)
        # plt.subplot(312)
        # plt.imshow(gt_np)
        # plt.subplot(313)
        # plt.imshow(depth_np)
        # plt.show()


        return rgb_np, depth_np, gt_np

    def val_transform(self, rgb, depth):
        depth = np.asfarray(depth, dtype='float32')
        rgb = transforms.functional.to_pil_image(rgb, mode=None)
        depth = transforms.functional.to_pil_image(depth, mode=None)
        # rgb = transforms.functional.crop(rgb, 140, 256, 224, 704)
        # depth = transforms.functional.crop(depth, 140, 256, 224, 704)
        rgb = transforms.functional.crop(rgb, 140, 10, 224, 1216)
        depth = transforms.functional.crop(depth, 140, 10, 224, 1216)
        # transform = transforms.Compose([
            # transforms.CenterCrop((352, 1216)), 
        # ])
        # rgb = transform(rgb)
        # depth = transform(depth)
        sample = UniformSampling(num_samples=self.samples)
        # sample = Drop_depth(prob_keep=0.3)
        rgb_np = rgb
        depth_np = depth
        rgb_np = np.asfarray(rgb_np, dtype='float') / 255
        depth_np = np.asfarray(depth_np, dtype='float32')
        gt_np = depth_np

        # Dense depth interpolation
        if ~self.sparse_gt:
            x, y = np.where(gt_np > 0)
            d = gt_np[gt_np != 0]
            xyd = np.stack((y, x, d)).T
            gt_np = lin_interp(gt_np.shape, xyd)
        gt_np = np.reshape(gt_np, (gt_np.shape[0], gt_np.shape[1], 1))

        depth_np = np.reshape(depth_np, (depth_np.shape[0], depth_np.shape[1], 1))
        depth_np = sample(depth_np)
        # depth_np = drop_depth_measurements(depth_np, 0.3)
        # plt.subplot(311)
        # plt.imshow(rgb_np)
        # plt.subplot(312)
        # plt.imshow(gt_np)
        # plt.subplot(313)
        # plt.imshow(depth_np)
        # plt.show()

        return rgb_np, depth_np, gt_np 
