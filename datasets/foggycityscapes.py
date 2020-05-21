import json
import os
import glob
from collections import namedtuple
import zipfile

from torchvision.datasets.utils import extract_archive, verify_str_arg, iterable_to_str
from .vision import VisionDataset
from .cityscapes import CityscapesInfo
from utils.data import StackDataset, ConcatDataset
from PIL import Image

class FoggyCityscapes__(VisionDataset, CityscapesInfo):
    """`Cityscapes <http://www.cityscapes-dataset.com/>`_ Dataset.
    Args:
        root (string): Root directory of dataset where directory ``leftImg8bit``
            and ``gtFine`` or ``gtCoarse`` are located.
        split (string, optional): The image split to use, ``train``, ``test`` or ``val`` if mode="fine"
            otherwise ``train``, ``train_extra`` or ``val``
        image_mode (string, optional): The image mode to use, ``leftImg8bit``, ``gtFine`` or ``gtCoarse``
        image_type (string, optional): Type of image to use, ``_instanceIds.png``, ``_labelIds.png``, 
            ``_color.png`` or ``_polygons.json``.
        image_transforms (list, callable, optional): A function/transform that takes sample as entry and
            returns a transformed version. E.g, ``transforms.RandomCrop``
    Examples:
        Get semantic segmentation target
        .. code-block:: python
            dataset = Cityscapes('./data/cityscapes', split='train', image_mode='gtFine',
                                 image_type='_labelIds.png')
            smnt = dataset[0]
        Validate on the "coarse" set
        .. code-block:: python
            dataset = Cityscapes('./data/cityscapes', split='val', mode='gtCoarse',
                                 image_type='_labelIds.png')
            smnt = dataset[0]
    """
    def __init__(self, root, split='train', # cities
                 image_mode='leftImg8bit', image_type='_leftImg8bit.png', image_transform=None,
                ):
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
            valid_types = ("_leftImg8bit.png",)
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
                
        self.root = root
        self.split = split
        self.image_mode = image_mode
        self.images_dir = os.path.join(image_mode, split)
        
        self.image_type = image_type
        self.image_transform = image_transform
        
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
        
        if 'Ids.png' in image_type:
            super(FoggyCityscapes__, self).__init__(root, self.images_dir, image_type, convert='L', transform=image_transform)
        elif '.json' in image_type and image_transform is None:
            super(FoggyCityscapes__, self).__init__(root, self.images_dir, image_type, transform=self._load_json)
        else:
            super(FoggyCityscapes__, self).__init__(root, self.images_dir, image_type, transform=image_transform)
        
        self.images = sorted([file_name for file_name
                              in glob.glob(os.path.join(self.root, self.images_dir, '*', '*{}'.format(self.image_type)))])
        
    def __repr__(self):
        head = "VisionDataset " + self.__class__.__name__
        body = ["Number of datapoints: {}".format(self.__len__())]
        if self.root is not None:
            body.append("Root location: {}".format(self.root))
        body += self.extra_repr().splitlines()
        if hasattr(self, "transforms") and self.transforms is not None:
            body += [repr(self.transforms)]
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return '\n'.join(lines)
    
    def extra_repr(self):
        lines = ["Split: {split}", "Mode: {image_mode}", "Type: {image_type}"]
        return '\n'.join(lines).format(**self.__dict__)
    
    def _load_json(self, path):
        with open(path, 'r') as file:
            data = json.load(file)
        return data
    
    
class FoggyCityscapes_(StackDataset, CityscapesInfo):
    _repr_indent = VisionDataset._repr_indent
    
    def __init__(self, root, split='train', # cities
                 image_mode='leftImg8bit', image_types=['_leftImg8bit.png'], image_transforms=None,
                ):
        self.root = root
        self.split = split
        self.image_mode = image_mode
        self.images_dir = os.path.join(image_mode, split)

        self.image_types = image_types
        
        if isinstance(image_transforms, list):
            assert len(image_transforms) == len(image_types), (
                "list image_transforms should have same length with image_types")
            self.image_transforms = image_transforms
        else:
            self.image_transforms = [image_transforms for _type in image_types]
        cityscapes__ = [FoggyCityscapes__(root, split, image_mode, image_type=_types, image_transform=_transform)
                        for _types, _transform in zip(self.image_types, self.image_transforms)] 
        super(FoggyCityscapes_, self).__init__(cityscapes__)

    def extra_repr(self):
        lines = ["Split: {split}", "Mode: {image_mode}", "Types: {image_types}"]
        return '\n'.join(lines).format(**self.__dict__)

    @property
    def images(self):
        return list(zip(*[dataset.images for dataset in self.datasets]))

    
class FoggyCityscapes(StackDataset, CityscapesInfo):
    _repr_indent = VisionDataset._repr_indent
    
    def __init__(self, root, split='train', # cities
                 image_modes=['leftImg8bit'], image_types=[['_leftImg8bit.png']], image_transforms=None,
                ):
        self.root = root
        self.split = split
        self.image_modes = image_modes
        self.image_types = image_types
        assert len(image_modes) == len(image_types), "image_modes and image_types should have same length"
        
        if isinstance(image_transforms, list):
            assert len(image_transforms) == len(image_modes), (
                "list image_transforms should have same length with image_modes")
            self.image_transforms = image_transforms
        else:
            self.image_transforms = [image_transforms for _type in image_modes]
            
        print(self.image_modes, self.image_types, self.image_transforms)
        cityscapes_ = [FoggyCityscapes_(root, split, _mode, _types, _transforms)
                       for _mode, _types, _transforms in zip(image_modes, image_types, self.image_transforms)]

        super(FoggyCityscapes, self).__init__(cityscapes_)
    
    def extra_repr(self):
        lines = ["Split: {split}", "Modes: {image_modes}", "Types: {image_types}"]
        return '\n'.join(lines).format(**self.__dict__)

    @property
    def images(self):
        return list(zip(*[dataset.images for dataset in self.datasets]))
