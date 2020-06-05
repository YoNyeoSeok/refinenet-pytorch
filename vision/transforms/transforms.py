import math
import random
import numpy as np
from PIL import Image

import torchvision.transforms as trf


class RandomHorizontalFlip(trf.RandomHorizontalFlip):
    def __init__(self, p):
        super(RandomHorizontalFlip, self).__init__(p)
        self.hflip = None
    def __call__(self, img):
        if self.hflip is None:
            self.update()
        return trf.functional.hflip(img) if self.hflip else img
    def update(self):
        self.hflip = random.random() < self.p

class RandomResizedCrop(trf.RandomResizedCrop):
    def __init__(self, size, scale, ratio, interpolation=Image.BILINEAR):
        super(RandomResizedCrop, self).__init__(size, scale, ratio, interpolation)
        self.img_size = None
    def __call__(self, img):
        if self.img_size is None:
            self.img_size = img.size #trf._get_image_size(img)
            self.update()
        return trf.functional.resized_crop(img, self.i, self.j, self.h, self.w, self.size, self.interpolation)
        
    @staticmethod
    def get_params(img_size, area_scale, aspect_ratio):
        width, height = img_size
        area = height * width

        target_area = area*area_scale
        w = int(round(math.sqrt(target_area * aspect_ratio)))
        h = int(round(math.sqrt(target_area / aspect_ratio)))

        if 0 < w <= width and 0 < h <= height:
            i = random.randint(0, height - h)
            j = random.randint(0, width - w)
            return i, j, h, w
        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if (in_ratio < aspect_ratio):
            w = width
            h = int(round(w / aspect_ratio))
        elif (in_ratio > aspect_ratio):
            h = height
            w = int(round(h * aspect_ratio))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w

    def update(self):
        self._scale = random.uniform(*self.scale)
        self._ratio = math.exp(random.uniform(*list(map(math.log, self.ratio))))
        crop_area_scale = np.prod([wh / img_wh * self._scale for wh, img_wh in zip(self.size, self.img_size)])
        self.i, self.j, self.h, self.w = self.get_params(self.img_size, crop_area_scale, self._ratio)
