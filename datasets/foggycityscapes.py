import json
import os
import glob
from collections import namedtuple
import zipfile

from torchvision.datasets.utils import extract_archive, verify_str_arg, iterable_to_str
from .vision import VisionDataset, ConcatVisionDataset, StackVisionDataset
from .cityscapes import CityscapesInfo, Cityscapes__, Cityscapes_, Cityscapes
from torch.utils.data import Subset
# from utils.data import StackDataset, ConcatDataset
from PIL import Image

class FoggyCityscapesInfo(CityscapesInfo):
    _repr_indent = VisionDataset._repr_indent

    def verify_mode(self, image_mode):
        valid_modes = ("gtFine", "gtFine",
                       "gtCoarse", "gtCoarse",
                       "leftImg8bit", "clear",
                       "leftImg8bit_foggy", "foggy",
                       "leftImg8bit_foggyDBF", "foggyDBF",
                       "leftImg8bit_transmittance", "transmittance",
                       "leftImg8bit_transmittanceDBF", "transmittanceDBF",)
        for i in range(len(valid_modes)//2):
            if image_mode == valid_modes[i*2+1]:
                image_mode = valid_modes[i*2]
                break
        verify_str_arg(image_mode, "image_mode", valid_modes)
        return image_mode
        
    def verify_mode_type(self, split, image_mode, image_type):
        image_mode = self.verify_mode(image_mode)
        
        if image_mode == "gtFine":
            valid_splits = ("train", "test", "val")
            valid_types = ("_instanceIds.png", "instance",
                           "_labelIds.png", "semantic",
                           "_color.png", "color",
                           "_polygons.json", "polygon")
        elif image_mode == "gtCoarse":
            valid_splits = ("train", "train_extra", "val")
            valid_types = ("_instanceIds.png", "instance",
                           "_labelIds.png", "semantic",
                           "_color.png", "color",
                           "_polygons.json", "polygon")
        elif image_mode == "leftImg8bit":
            valid_splits = ("train", "train_extra", "test", "val")
            valid_types = ("_leftImg8bit.png", None,)
        elif image_mode == "leftImg8bit_foggy" or image_mode == "leftImg8bit_foggyDBF":
            valid_splits = ("train", "train_extra", "test", "val")
            valid_types = ("_leftImg8bit_foggy_beta_0.005.png", "beta_0.005",
                           "_leftImg8bit_foggy_beta_0.01.png", "beta_0.01",
                           "_leftImg8bit_foggy_beta_0.02.png", "beta_0.02",)
        elif image_mode == "leftImg8bit_transmittance" or image_mode == "leftImg8bit_transmittanceDBF":
            valid_splits = ("train", "train_extra", "test", "val")
            valid_types = ("_leftImg8bit_transmittance_beta_0.005.png", "beta_0.005",
                           "_leftImg8bit_transmittance_beta_0.01.png", "beta_0.01",
                           "_leftImg8bit_transmittance_beta_0.02.png", "beta_0.02",)
        for i in range(len(valid_types)//2):
            if image_type == valid_types[i*2+1]:
                image_type = valid_types[i*2]
                break
        
        msg = ("Unknown value '{}' for argument split if image_mode is '{}'. "
               "Valid values are {{{}}}.")
        msg = msg.format(split, image_mode, iterable_to_str(valid_splits))
        verify_str_arg(split, "split", valid_splits, msg)
        
        msg = ("Unknown value '{}' for argument image_type if image_mode is '{}'. "
               "Valid values are {{{}}}.")
        msg = msg.format(image_type, image_mode, iterable_to_str(valid_types))
        verify_str_arg(image_type, "image_type", valid_types, msg)
        return image_mode, image_type
    
    def verify_dataset(self):
        if not os.path.isdir(os.path.join(self.root, self.images_dir)):
            if self.image_mode == 'gtFine':
                image_dir_zip = os.path.join(self.root, '{}_trainvaltest.zip'.format(self.image_mode))
            elif self.image_mode == 'gtCoarse':
                image_dir_zip = os.path.join(self.root, '{}.zip'.format(self.image_mode))
            else:
                if split == 'train_extra':
                    if self.image_mode == 'leftImg8bit':
                        image_dir_zip = os.path.join(self.root, '{}_trainextra.zip'.format(self.image_mode))
                    else:
                        image_dir_zip = os.path.join(self.root, '{}_trainextra_{}.zip'.format(*self.image_mode.split('_')))
                else:
                    if self.image_mode == 'leftImg8bit':
                        image_dir_zip = os.path.join(self.root, '{}_trainvaltest.zip'.format(self.image_mode))
                    else:
                        image_dir_zip = os.path.join(self.root, '{}_trainvaltest_{}.zip'.format(*self.image_mode.split('_')))
                
            if os.path.isfile(image_dir_zip):
                extract_archive(from_path=image_dir_zip, to_path=self.root)
                extract_archive(from_path=target_dir_zip, to_path=self.root)
            else:
                raise RuntimeError('Dataset not found or incomplete. Please make sure all required folders for the'
                                   ' specified "split" and "image_mode" are inside the "root" directory')
        

class FoggyCityscapes__(Cityscapes__, FoggyCityscapesInfo): pass
    
    
class FoggyCityscapes_(FoggyCityscapesInfo, Cityscapes_):
    Cityscapes__ = FoggyCityscapes__


class FoggyCityscapes(FoggyCityscapesInfo, Cityscapes):
    Cityscapes_ = FoggyCityscapes_

    
class RefinedFoggyCityscapes(Subset, FoggyCityscapesInfo):
    __repr__ = FoggyCityscapes.__repr__
    _format_transform_repr = VisionDataset._format_transform_repr
    extra_repr = FoggyCityscapes.extra_repr
    
    def __init__(self, root, split='train', # cities
                 image_modes=['leftImg8bit'], image_types=[['_leftImg8bit.png']], image_transforms=None,
                 refined_filenames='foggy_trainval_refined_filenames.txt'):
        cityscapes = FoggyCityscapes(root, split, image_modes, image_types, image_transforms)
        self.root = cityscapes.root
        self.split = cityscapes.split
        self.image_modes = cityscapes.image_modes
        self.image_types = cityscapes.image_types
        self.image_transforms = cityscapes.image_transforms

        self.refined_filenames = refined_filenames
        
        if os.path.isfile(os.path.join(root, refined_filenames)):
            with open(os.path.join(root, refined_filenames)) as f:
                lines = list(map(str.strip, f.readlines()))            
            indices = [a for a, ((img, ), _) in enumerate(cityscapes.images) for line in sorted(lines) if line in img]
        else:
            raise RuntimeError('refined_filenames are not found or incomplete. Please make sure the required file'
                                'is inside the "root" directory')

        super(RefinedFoggyCityscapes, self).__init__(cityscapes, indices)

    @property
    def images(self):
        return [self.dataset.images[idx] for idx in self.indices]