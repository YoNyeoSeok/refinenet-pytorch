import json
import os
import glob
from collections import namedtuple
import zipfile

from torchvision.datasets.utils import extract_archive, verify_str_arg, iterable_to_str
from .vision import VisionDataset, ConcatVisionDataset, StackVisionDataset
from .cityscapes import CityscapesInfo
from torch.utils.data import Subset
# from utils.data import StackDataset, ConcatDataset
from PIL import Image


class FoggyCityscapes__(ConcatVisionDataset, CityscapesInfo):
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
                 image_mode='leftImg8bit', image_type='_leftImg8bit.png', image_transforms=None,
                ):
        self.root = root
        self.split = split

        image_mode, image_type = self.verify_mode_type(split, image_mode, image_type)
        self.image_mode = image_mode
        self.image_type = image_type
        
        self.image_transforms = image_transforms
        
        self.images_dir = os.path.join(image_mode, split)
        self.verify_dataset()
        
        convert = 'L' if 'Ids.png' in image_type else 'RGB'
        if '.json' in image_type and transform is None:
            image_transforms = transforms.Lambda(lambda x: self._load_json(x))
                
        dataset = [VisionDataset(root, os.path.join(self.images_dir, city), image_type, convert, image_transforms) 
                   for city in sorted(glob.glob(os.path.join(root, self.images_dir, '*')))]
        super(FoggyCityscapes__, self).__init__(dataset)
        
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
        
    def extra_repr(self):
        lines = ["Split: {split}", "Mode: {image_mode}", "Type: {image_type}"]
        return '\n'.join(lines).format(**self.__dict__)
    
    
class FoggyCityscapes_(StackVisionDataset, CityscapesInfo):
    _repr_indent = VisionDataset._repr_indent
    verify_mode = FoggyCityscapes__.verify_mode
    verify_mode_type = FoggyCityscapes__.verify_mode_type
    verify_dataset = FoggyCityscapes__.verify_dataset
    
    def __init__(self, root, split='train', # cities
                 image_mode='leftImg8bit', image_types=['_leftImg8bit.png'], image_transforms=None,
                ):
        self.root = root
        self.split = split

        image_mode = self.verify_mode(image_mode)
        _, image_types = list(zip(*[self.verify_mode_type(split, image_mode, image_type) for image_type in image_types]))
        self.image_mode = image_mode
        self.image_types = image_types
        
        image_transforms = (image_transforms 
                            if isinstance(image_transforms, list)
                            else [image_transforms for _type in image_types])
        assert len(image_transforms) == len(image_types), (
            "list image_transforms should have same length with image_types")
        self.image_transforms = image_transforms
        
        self.images_dir = os.path.join(image_mode, split)
        self.verify_dataset()
        
        dataset = [FoggyCityscapes__(root, split, image_mode, image_type, image_transforms)
                   for image_type, image_transforms in zip(self.image_types, self.image_transforms)] 
        super(FoggyCityscapes_, self).__init__(dataset)
        
    def extra_repr(self):
        lines = ["Split: {split}", "Mode: {image_mode}", "Types: {image_types}"]
        return '\n'.join(lines).format(**self.__dict__)

    @property
    def images(self):
        return list(zip(*[dataset.images for dataset in self.datasets]))

    
class FoggyCityscapes(StackVisionDataset, CityscapesInfo):
    _repr_indent = VisionDataset._repr_indent
    verify_mode = FoggyCityscapes_.verify_mode
    verify_mode_type = FoggyCityscapes_.verify_mode_type
    verify_dataset = FoggyCityscapes_.verify_dataset
    
    def __init__(self, root, split='train', # cities
                 image_modes=['leftImg8bit'], image_types=[['_leftImg8bit.png']], image_transforms=None,
                ):
        self.root = root
        self.split = split

        image_modes = [self.verify_mode(image_mode) for image_mode in image_modes]
        _, image_types = list(zip(*[    list(zip(*[self.verify_mode_type(split, image_mode, image_type) 
                                                   for image_type in _types]))
                                    for image_mode, _types in zip(image_modes, image_types)]))
        assert len(image_modes) == len(image_types), "image_modes and image_types should have same length"
        self.image_modes = image_modes
        self.image_types = image_types

        image_transforms = (image_transforms 
                            if isinstance(image_transforms, list)
                            else [image_transforms for _mode in image_modes])
        assert len(image_transforms) == len(image_types), (
            "list image_transforms should have same length with image_types")
        self.image_transforms = image_transforms
            
        cityscapes_ = [FoggyCityscapes_(root, split, _mode, _types, _transforms)
                       for _mode, _types, _transforms in zip(image_modes, image_types, self.image_transforms)]

        super(FoggyCityscapes, self).__init__(cityscapes_)
    
    def extra_repr(self):
        lines = ["Split: {split}", "Modes: {image_modes}", "Types: {image_types}"]
        return '\n'.join(lines).format(**self.__dict__)

    @property
    def images(self):
        return list(zip(*[dataset.images for dataset in self.datasets]))

    
class RefinedFoggyCityscapes(Subset, CityscapesInfo):
    _repr_indent = VisionDataset._repr_indent
    verify_mode = FoggyCityscapes.verify_mode
    verify_mode_type = FoggyCityscapes.verify_mode_type
    verify_dataset = FoggyCityscapes.verify_dataset
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