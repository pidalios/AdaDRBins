"""
Change log:
    2022:
    8/1: Apply Gaussian noise to median pooling
    8/5: Add Sparse sampling function
    8/16: Add RandomSubblockSampling
"""

import numpy as np
import cv2
import torch

"""
All Inputs should be ndarray with shape (H, W, C)
"""

# class OffestMedianPooling(object):
    # def __init__(self, p=8, sigma=0.01):
        # self.p = p
        # self.sigma = sigma
        # self.size = np.lcm(224, p)

def drop_depth_measurements(depth, prob_keep):
    # print(np.count_nonzero(depth))
    mask = np.random.binomial(1, prob_keep, depth.shape)
    out = depth * mask
    # print(np.count_nonzero(depth))
    # print('--')
    return out

class Drop_depth(object):
    def __init__(self, prob_keep=0.3):
        self.prob_keep = prob_keep
    def __call__(self, depth):
        mask = np.random.binomial(1, self.prob_keep, depth.shape)
        depth = depth * mask
        # print(np.count_nonzero(depth))
        # print('--')
        return depth


class MedianPooling(object):
    """
    Input: ndarray (H, W, C)
    Output: ndarray (p, p, C)
    """
    def __init__(self, p=8, min_depth=0, max_depth=np.inf, sigma=0.01):
        self.p = p
        self.sigma = sigma
        self.size = np.lcm(224, p)
        self.min_depth = min_depth
        self.max_depth = max_depth

    def __call__(self, img):
        n = self.sigma*np.random.randn(self.p, self.p, 1)
        img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        img = np.expand_dims(img, axis=2)

        M = img.shape[0]//self.p
        subblocks = np.array([img[x:x+M, y:y+M, :] for x in range(0, img.shape[0], M) for y in range(0, img.shape[1], M)])
        m = np.zeros((self.p**2, img.shape[2]))
        for i in range(self.p**2):
            subblock = subblocks[i, :, :, :]
            mask = subblock > 0
            masked_subblock = subblock[mask]
            if len(masked_subblock)==0:
                m[i, :] = 0
            else:
                depth = np.median(masked_subblock)
                if depth < self.min_depth or depth > self.max_depth:
                    depth = 0
                m[i, :] = depth
            median = np.zeros((self.p, self.p, img.shape[2]))
        for x in range(self.p):
            for y in range(self.p):
                median[x, y, :] = m[self.p*x+y, :]

        # Apply Gaussian noise
        median = median + n
        return median

class RandomSparseSampling(object):
    """
    Input: ndarray (H, W, C)
    Output: ndarray (H, W, C)
    """
    def __init__(self, n=200):
        self.n = n
    
    def __call__(self, img):
        H, W, C = img.shape
        valid_mask = img > 0
        img_masked = img[valid_mask]
        if img_masked.shape[0] == 0:
            prob = self.n/(H*W)
        else:
            prob = self.n/img_masked.shape[0]
        
        output = np.zeros_like(img)
        for h in range(H):
            for w in range(W):
                occur = np.random.uniform(0, 1) < prob
                if occur and valid_mask[h, w]:
                    output[h, w, :] = img[h, w, :]
        return output

class RandomSubblockSampling(object):
    """
    Input: ndarray (H, W, C)
    Output: ndarray (p, p, C)
    """
    def __init__(self, p=8, sigma=0.02):
        self.p = p
        self.size = np.lcm(224, p)
        self.sigma = sigma

    def __call__(self, img):
        n = self.sigma*np.random.randn(self.p, self.p, 1)
        img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        img = np.expand_dims(img, axis=2)

        M = img.shape[0]//self.p
        subblocks = np.array([img[x:x+M, y:y+M, :] for x in range(0, img.shape[0], M) for y in range(0, img.shape[1], M)])
        m = np.zeros((self.p**2, img.shape[2]))
        for i in range(self.p**2):
            x = np.random.randint(0, M)
            y = np.random.randint(0, M)
            m[i, :] = subblocks[i, x, y, :]
        sample = np.zeros(self.p, self.p, 1)
        for x in range(self.p):
            for y in range(self.p):
                sample[x, y, :] = m[self.p*x+y, :]

        # Apply Gaussian noise
        sample = sample + n
        return sample

class UniformSampling(object):
    name = "uar"
    def __init__(self, num_samples, max_depth=np.inf):
        # DenseToSparse.__init__(self)
        self.num_samples = num_samples
        self.max_depth = max_depth

    def __repr__(self):
        return "%s{ns=%d,md=%f}" % (self.name, self.num_samples, self.max_depth)

    def __call__(self, depth):
        """
        Samples pixels with `num_samples`/#pixels probability in `depth`.
        Only pixels with a maximum depth of `max_depth` are considered.
        If no `max_depth` is given, samples in all pixels
        """
        mask_keep = depth > 0
        if self.max_depth is not np.inf:
            mask_keep = np.bitwise_and(mask_keep, depth <= self.max_depth)
        n_keep = np.count_nonzero(mask_keep)
        if n_keep == 0:
            depth_sp = np.zeros_like(depth)
            return depth_sp
        else:
            prob = float(self.num_samples) / n_keep
            mask = np.bitwise_and(mask_keep, np.random.uniform(0, 1, depth.shape) < prob)
            depth_sp = np.zeros_like(depth)
            depth_sp[mask] = depth[mask]
            return depth_sp
class ToTensor(object):
    """
    Input: ndarray (H, W, C)
    Output: torch.Tensor (C, H, W)
    """
    def __call__(self, img):
        # if img.ndim == 2:
            # h, w = img.shape
            # img = img.reshape(h, w, 1)
        img = np.transpose(img, (2, 0, 1))
        img = torch.tensor(img)
        return img.float()

if __name__=='__main__':
    sparseSampling = RandomSparseSampling(n=10)
    img = np.random.randn(8, 8, 1)
    output = sparseSampling(img)
    print(img.shape)
    print(img)
    print(output.shape)
    print(output)
