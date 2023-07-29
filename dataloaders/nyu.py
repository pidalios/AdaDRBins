import numpy as np
import torch
import dataloaders.trans as trans
from dataloaders.dataloader import MyDataloader
# import trans
# from dataloader import MyDataloader
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import os
import matplotlib.pyplot as plt
from scipy.interpolate import LinearNDInterpolator

iheight, iwidth = 480, 640 # raw image size

def lin_interp(shape, xyd):
    # taken from https://github.com/hunse/kitti
    m, n = shape
    ij, d = xyd[:, 1::-1], xyd[:, 2]
    f = LinearNDInterpolator(ij, d, fill_value=0)
    J, I = np.meshgrid(np.arange(n), np.arange(m))
    IJ = np.vstack([I.flatten(), J.flatten()]).T
    disparity = f(IJ).reshape(shape)
    return disparity

class NYUDataset(MyDataloader):
    def __init__(self, root, split, min_depth=1e-3, max_depth=np.inf, lowRes=True, num_samples=500):
        super(NYUDataset, self).__init__(root, split)
        self.split = split
        self.lowRes = lowRes
        self.num_samples = num_samples
        self.min_depth = min_depth
        self.max_depth = max_depth
        if lowRes:
            self.output_size = (224, 224)
        else:
            self.output_size = (228, 304)

    def train_transform(self, rgb, depth):
        """
        Input is ndarray with shape (H, W, C)
        Output is an ndarray with shape (H, W, C)
        """
        s = np.random.uniform(1.0, 1.5) # random scaling
        angle = np.random.uniform(-5.0, 5.0) # random rotation degrees
        do_flip = np.random.uniform(0.0, 1.0) < 0.5 # random horizontal flip

        transform = transforms.Compose([
            transforms.ToPILImage(), 
            transforms.RandomRotation((angle, angle), interpolation=InterpolationMode.BILINEAR), 
            transforms.Resize((int(np.ceil(360/s)), int(np.ceil(480/s))), InterpolationMode.BILINEAR), 
            transforms.RandomHorizontalFlip(do_flip), 
            transforms.CenterCrop(self.output_size), 
            ])
        if self.lowRes:
            sample = trans.MedianPooling(p=8, min_depth=self.min_depth, max_depth=self.max_depth, sigma=0)
        else:
            sample = trans.UniformSampling(num_samples=self.num_samples)
        color_jitter = transforms.ColorJitter(0.4, 0.4, 0.4)

        rgb = transform(rgb)
        rgb = color_jitter(rgb) # random color jittering
        depth = transform(depth)
        rgb_np = np.asfarray(rgb, dtype='float') / 255
        depth_np = np.array(depth)  # shape H, W, C
        depth_gt_np = depth_np
        depth_np = sample(depth_np)
        # print(depth_np.shape)

        if self.lowRes == False:
            # print(depth_np.shape)
            # x, y = np.where(depth_np > 0)
            # d = depth_np[depth_np != 0]
            # xyd = np.stack((y, x, d)).T
            # depth_np = lin_interp(depth_np.shape, xyd)

            depth_np = np.pad(depth_np, ((14, 14), (8, 8)))
            depth_gt_np = np.pad(depth_gt_np, ((14, 14), (8, 8)))
            rgb_np = np.asarray([np.pad(rgb_np[:, :, c], ((14, 14), (8, 8))) for c in range(3)])
            rgb_np = np.transpose(rgb_np, (1, 2, 0))


        depth_np = np.reshape(depth_np, (depth_np.shape[0], depth_np.shape[1], 1))
        depth_gt_np = np.reshape(depth_gt_np, (depth_gt_np.shape[0], depth_gt_np.shape[1], 1))

        # depth_np = medianpool(depth_np)
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
        transform = transforms.Compose([
            transforms.ToPILImage(), 
            transforms.Resize((264, 352)), 
            transforms.CenterCrop(self.output_size), 
            ])
        # sparseSample = trans.RandomSparseSampling(n=500)
        if self.lowRes:
            sample = trans.MedianPooling(p=8, min_depth=self.min_depth, max_depth=self.max_depth, sigma=0)
        else:
            sample = trans.UniformSampling(num_samples=self.num_samples)


        rgb = transform(rgb)
        depth = transform(depth)
        rgb_np = np.asfarray(rgb, dtype='float') / 255
        depth_np = np.array(depth)  # shape H, W, C
        depth_gt_np = depth_np
        # depth_gt_np = np.array(depth)
        # depth_gt_np = np.reshape(depth_gt_np, (depth_np.shape[0], depth_np.shape[1], 1))

        # depth_np = medianpool(depth_np)
        depth_np = sample(depth_np)
        if self.lowRes == False:
            # print(depth_np.shape)
            depth_np = np.pad(depth_np, ((14, 14), (8, 8)))
            depth_gt_np = np.pad(depth_gt_np, ((14, 14), (8, 8)))
            rgb_np = np.asarray([np.pad(rgb_np[:, :, c], ((14, 14), (8, 8))) for c in range(3)])
            rgb_np = np.transpose(rgb_np, (1, 2, 0))
            # x, y = np.where(depth_np > 0)
            # d = depth_np[depth_np != 0]
            # xyd = np.stack((y, x, d)).T
            # depth_np = lin_interp(depth_np.shape, xyd)
        
        depth_np = np.reshape(depth_np, (depth_np.shape[0], depth_np.shape[1], 1))
        depth_gt_np = np.reshape(depth_gt_np, (depth_gt_np.shape[0], depth_gt_np.shape[1], 1))
        # print(depth_gt_np.shape)

        return rgb_np, depth_np, depth_gt_np

if __name__=='__main__':
    print('Load data...')
    traindir = os.path.join('..', 'data', 'nyudepthv2', 'train')
    train_dataset = NYUDataset(traindir, split='train', modality='rgb')
    rgb, depth, gt = train_dataset.__getitem__(1800)


    
# import torch
# import dataloaders.trans as trans
# from dataloaders.dataloader import MyDataloader
# # import trans
# # from dataloader import MyDataloader
# from torchvision import transforms
# from torchvision.transforms.functional import InterpolationMode
# import os
# import matplotlib.pyplot as plt
# import random
# from PIL import Image

# iheight, iwidth = 480, 640 # raw image size

# class NYUDataset(MyDataloader):
#     def __init__(self,
#                  root,
#                  split,
#                  annotation_path=None, 
#                  hdf5=True, 
#                  lowRes=1, 
#                  num_samples=500,
#                  min_depth=1e-3,
#                  max_depth=np.inf):
#         self.split = split
#         super(NYUDataset, self).__init__(root, split, annotation_path, hdf5)
#         self.output_size = (224, 224)
#         self.lowRes = lowRes
#         self.min_depth = min_depth
#         self.max_depth = max_depth
#         self.num_sample = num_samples

#     def random_crop(self, img, depth, height, width):
#         assert img.shape[0] >= height
#         assert img.shape[1] >= width
#         assert img.shape[0] == depth.shape[0]
#         assert img.shape[1] == depth.shape[1]
#         x = random.randint(0, img.shape[1] - width)
#         y = random.randint(0, img.shape[0] - height)
#         img = img[y:y + height, x:x + width, :]
#         depth = depth[y:y + height, x:x + width]
#         return img, depth
    
#     def train_preprocess(self, image, depth_gt):
#         # Random flipping
#         do_flip = random.random()
#         if do_flip > 0.5:
#             image = (image[:, ::-1, :]).copy()
#             depth_gt = (depth_gt[:, ::-1]).copy()
    
#         # Random gamma, brightness, color augmentation
#         do_augment = random.random()
#         if do_augment > 0.5:
#             image = self.augment_image(image)
    
#         return image, depth_gt
    
#     def augment_image(self, image):
#         # gamma augmentation
#         gamma = random.uniform(0.9, 1.1)
#         image_aug = image ** gamma

#         # brightness augmentation
#         brightness = random.uniform(0.75, 1.25)
#         image_aug = image_aug * brightness
#         return image_aug
    
#     def random_rotate(self, image, depth_gt, angle, flag=Image.BILINEAR):
#         """
#         Input is a PIL image
#         """
#         random_angle = np.random.uniform(-angle, angle)
#         image = image.rotate(random_angle, resample=flag)
#         depth_gt = depth_gt.rotate(random_angle, resample=flag)
#         return image, depth_gt

#     def train_transform(self, rgb, depth):
#         """
#         Input is ndarray with shape (480, 640, 3)
#         Output is an ndarray with shape (480, 640, 1)
#         """

#         s = np.random.uniform(1.0, 1.5) # random scaling
#         angle = np.random.uniform(-5.0, 5.0) # random rotation degrees
#         do_flip = np.random.uniform(0.0, 1.0) < 0.5 # random horizontal flip

#         transform = transforms.Compose([
#             transforms.ToPILImage(), 
#             transforms.Resize((240, 320), interpolation=InterpolationMode.BILINEAR)
#             ])
        
#         medianpool = trans.MedianPooling(p=8, sigma=0, min_depth=self.min_depth, max_depth=self.max_depth)
#         sparseSample = trans.UniformSampling(num_samples=self.num_sample)
        
#         rgb = transform(rgb)
#         depth = transform(depth)
#         rgb, depth = self.random_rotate(rgb, depth, 5)
#         rgb_np = np.asfarray(rgb, dtype='float') / 255.0
#         depth_np = np.asfarray(depth, dtype='float')  # shape H, W, C
#         depth_gt_np = np.asfarray(depth, dtype='float')
#         rgb_np, depth_np = self.random_crop(rgb_np, depth_np, 224, 224)
#         rgb_np, depth_np = self.train_preprocess(rgb_np, depth_np)
#         depth_np = np.reshape(depth_np, (depth_np.shape[0], depth_np.shape[1], 1))
#         depth_gt_np = depth_np

#         if self.lowRes == 1:
#             depth_np = medianpool(depth_np)
#         else:
#             depth_np = sparseSample(depth_np)

#         depth_np = depth_np/10
#         # print(type(rgb_np))
#         # plt.subplot(131)
#         # plt.imshow(rgb_np)
#         # plt.subplot(132)
#         # plt.imshow(depth_np)
#         # plt.subplot(133)
#         # plt.imshow(depth_gt_np)
#         # plt.show()

#         return rgb_np, depth_np, depth_gt_np

#     def val_transform(self, rgb, depth):
#         """
#         Input is an ndarray with shape (H, W, C)
#         Output is an ndarray with shape (H, W, C)
#         """
#         transform = transforms.Compose([
#             transforms.ToPILImage(), 
#             transforms.Resize((240, 320), interpolation=InterpolationMode.BILINEAR),
#             transforms.CenterCrop(self.output_size)
#             ])

#         sparseSample = trans.UniformSampling(num_samples=500)
#         medianpool = trans.MedianPooling(p=8, sigma=0, min_depth=self.min_depth, max_depth=self.max_depth)

#         rgb = transform(rgb)
#         depth = transform(depth)
#         rgb_np = np.asfarray(rgb, dtype='float') / 255
#         depth_np = np.array(depth)  # shape H, W, C
#         depth_gt_np = np.array(depth)
#         depth_np = np.reshape(depth_np, (depth_np.shape[0], depth_np.shape[1], 1))
#         depth_gt_np = depth_np

#         if self.lowRes == 1:
#             depth_np = medianpool(depth_np)
#         else:
#             depth_np = sparseSample(depth_np)

#         # Normalize depth values to 0 - 1 to fit Unity textures format
#         depth_np = depth_np/10
#         # print(type(rgb_np))
#         # plt.subplot(131)
#         # plt.imshow(rgb_np)
#         # plt.subplot(132)
#         # plt.imshow(depth_np)
#         # plt.subplot(133)
#         # plt.imshow(depth_gt_np)
#         # plt.show()


#         return rgb_np, depth_np, depth_gt_np
# if __name__=='__main__':
#     print('Load data...')
#     traindir = os.path.join('..', 'data', 'nyudepthv2', 'train')
#     train_dataset = NYUDataset(traindir, split='train', modality='rgb')
#     rgb, depth, gt = train_dataset.__getitem__(1800)
