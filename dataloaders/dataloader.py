import os
import numpy as np
import cv2
import torch.utils.data as data
import h5py
import dataloaders.trans as trans
from torchvision import transforms
import torch.nn.functional as F
import torch.nn as nn
import torch
from PIL import Image
import json

def h5_loader(path):
    h5f = h5py.File(path, "r")
    rgb = np.array(h5f['rgb'])
    rgb = np.transpose(rgb, (1, 2, 0))
    depth = np.array(h5f['depth'])
    depth = np.reshape(depth, (depth.shape[0], depth.shape[1], 1))
    return rgb, depth

def jpg_loader(path):
    rgb = cv2.imread(path[0])
    depth = cv2.imread(path[1])
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    rgb = np.array(rgb)
    depth = np.array(depth)
    depth = np.reshape(depth[:, :, 0], (depth.shape[0], depth.shape[1], 1))
    return rgb, depth

def load_annotation_data(data_file_path):
    with open(data_file_path, 'r') as data_file:
        return json.load(data_file)

def get_video_names_and_annotations(data, subset):
    video_names = []
    annotations = []

    for key, value in data['database'].items():
        this_subset = value['subset']
        if this_subset in subset:
            label = value['annotations']['label']
            video_names.append(key.split('_')[0])
            annotations.append(value['annotations'])
    return video_names, annotations

class MyDataloader(data.Dataset):
    modality_names = ['rgb']

    def is_image_file(self, filename):
        IMG_EXTENSIONS = ['.h5']
        return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

    def find_classes(self, dir, annotation_path=None, split=None):
        if annotation_path != None:
            data = load_annotation_data(annotation_path)
            video_names, annotations = get_video_names_and_annotations(data, split)
            classes = video_names
            classes.sort()
            class_to_idx = {classes[i]: i for i in range(len(classes))}
        else:
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
            classes.sort()
            class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def make_dataset(self, dir, class_to_idx, hdf5=True, annotation_path=None, split=None):
        if hdf5 == True:
            images = []
            dir = os.path.expanduser(dir)
            for target in sorted(os.listdir(dir)):
                # print(target)
                d = os.path.join(dir, target)
                # print(d)
                if not os.path.isdir(d):
                    continue
                for root, _, fnames in sorted(os.walk(d)):
                    for fname in sorted(fnames):
                        if self.is_image_file(fname):
                            path = os.path.join(root, fname)
                            item = (path, class_to_idx[target])
                            images.append(item)
        else:
            images = []
            i = 0
            if annotation_path == None:
                raise ValueError('Annotation path must be specfied.')
            elif split == None:
                raise ValueError('Split must be specified.')
            data = load_annotation_data(annotation_path)
            video_names, annotations = get_video_names_and_annotations(data, split)

            for target in video_names:
                rgb_dir = os.path.join(dir, target)
                depth_dir = os.path.join(rgb_dir.rsplit(os.sep,2)[0] , 'Depth','depth' + rgb_dir[-1])
                if not os.path.isdir(rgb_dir):
                    continue
                for fname in sorted(os.listdir(rgb_dir)):
                    r = os.path.join(rgb_dir, fname)
                    d = os.path.join(depth_dir, fname)
                    if not os.path.exists(d):
                        continue
                    item = ([r, d], class_to_idx[target])
                    images.append(item)
        return images

    def __init__(self,
                 root,
                 split,
                 annotation_path=None, 
                 hdf5=True):
        if hdf5 == True:
            loader = h5_loader
        else:
            loader = jpg_loader
        classes, class_to_idx = self.find_classes(root, annotation_path, split)
        imgs = self.make_dataset(root, class_to_idx, hdf5=hdf5, annotation_path=annotation_path, split=split)
        assert len(imgs)>0, "Found 0 images in subfolders of: " + root + "\n"
        self.root = root
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.split = split
        if split == 'training':
            self.transform = self.train_transform
        elif split == 'validation':
            self.transform = self.val_transform
        self.loader = loader

    def __getraw__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (rgb, depth) the raw data.
        """
        path, target = self.imgs[index]
        rgb, depth = self.loader(path)
        return rgb, depth

    def __getitem__(self, index):
        rgb, depth = self.__getraw__(index)         # shape H, W, C
        rgb_np, depth_np, depth_gt_np= self.transform(rgb, depth)

        to_tensor = trans.ToTensor()

        input_rgb_tensor = to_tensor(rgb_np)
        input_depth_tensor = to_tensor(depth_np)
        gt_tensor = to_tensor(depth_gt_np)

        model_inputs = {"rgb": input_rgb_tensor,
                        "depth": input_depth_tensor,
                        "gt": gt_tensor}
        return model_inputs 

    def __len__(self):
        return len(self.imgs)

