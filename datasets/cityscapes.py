import json
import os
import glob
from collections import namedtuple
import zipfile

from torchvision.datasets.utils import extract_archive, verify_str_arg, iterable_to_str
from .vision import VisionDataset, ConcatVisionDataset, StackVisionDataset
# from utils.data import StackDataset, ConcatDataset
from PIL import Image

class CityscapesInfo:
    _repr_indent = VisionDataset._repr_indent
    # Based on https://github.com/mcordts/cityscapesScripts
    CityscapesClass = namedtuple('CityscapesClass', ['name', 'id', 'train_id', 'category', 'category_id',
                                                     'has_instances', 'ignore_in_eval', 'color'])

    classes = [
        CityscapesClass('unlabeled', 0, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('ego vehicle', 1, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('rectification border', 2, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('out of roi', 3, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('static', 4, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('dynamic', 5, 255, 'void', 0, False, True, (111, 74, 0)),
        CityscapesClass('ground', 6, 255, 'void', 0, False, True, (81, 0, 81)),
        CityscapesClass('road', 7, 0, 'flat', 1, False, False, (128, 64, 128)),
        CityscapesClass('sidewalk', 8, 1, 'flat', 1, False, False, (244, 35, 232)),
        CityscapesClass('parking', 9, 255, 'flat', 1, False, True, (250, 170, 160)),
        CityscapesClass('rail track', 10, 255, 'flat', 1, False, True, (230, 150, 140)),
        CityscapesClass('building', 11, 2, 'construction', 2, False, False, (70, 70, 70)),
        CityscapesClass('wall', 12, 3, 'construction', 2, False, False, (102, 102, 156)),
        CityscapesClass('fence', 13, 4, 'construction', 2, False, False, (190, 153, 153)),
        CityscapesClass('guard rail', 14, 255, 'construction', 2, False, True, (180, 165, 180)),
        CityscapesClass('bridge', 15, 255, 'construction', 2, False, True, (150, 100, 100)),
        CityscapesClass('tunnel', 16, 255, 'construction', 2, False, True, (150, 120, 90)),
        CityscapesClass('pole', 17, 5, 'object', 3, False, False, (153, 153, 153)),
        CityscapesClass('polegroup', 18, 255, 'object', 3, False, True, (153, 153, 153)),
        CityscapesClass('traffic light', 19, 6, 'object', 3, False, False, (250, 170, 30)),
        CityscapesClass('traffic sign', 20, 7, 'object', 3, False, False, (220, 220, 0)),
        CityscapesClass('vegetation', 21, 8, 'nature', 4, False, False, (107, 142, 35)),
        CityscapesClass('terrain', 22, 9, 'nature', 4, False, False, (152, 251, 152)),
        CityscapesClass('sky', 23, 10, 'sky', 5, False, False, (70, 130, 180)),
        CityscapesClass('person', 24, 11, 'human', 6, True, False, (220, 20, 60)),
        CityscapesClass('rider', 25, 12, 'human', 6, True, False, (255, 0, 0)),
        CityscapesClass('car', 26, 13, 'vehicle', 7, True, False, (0, 0, 142)),
        CityscapesClass('truck', 27, 14, 'vehicle', 7, True, False, (0, 0, 70)),
        CityscapesClass('bus', 28, 15, 'vehicle', 7, True, False, (0, 60, 100)),
        CityscapesClass('caravan', 29, 255, 'vehicle', 7, True, True, (0, 0, 90)),
        CityscapesClass('trailer', 30, 255, 'vehicle', 7, True, True, (0, 0, 110)),
        CityscapesClass('train', 31, 16, 'vehicle', 7, True, False, (0, 80, 100)),
        CityscapesClass('motorcycle', 32, 17, 'vehicle', 7, True, False, (0, 0, 230)),
        CityscapesClass('bicycle', 33, 18, 'vehicle', 7, True, False, (119, 11, 32)),
        CityscapesClass('license plate', -1, -1, 'vehicle', 7, False, True, (0, 0, 142)),
    ]

    def verify_mode(self, image_mode):
        valid_modes = ("gtFine", "gtFine",
                       "gtCoarse", "gtCoarse",
                       "leftImg8bit", None,)
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
                    image_dir_zip = os.path.join(self.root, '{}_trainextra.zip'.format(self.image_mode))
                else:
                    image_dir_zip = os.path.join(self.root, '{}_trainvaltest.zip'.format(self.image_mode))
                
            if os.path.isfile(image_dir_zip):
                extract_archive(from_path=image_dir_zip, to_path=self.root)
                extract_archive(from_path=target_dir_zip, to_path=self.root)
            else:
                raise RuntimeError('Dataset not found or incomplete. Please make sure all required folders for the'
                                   ' specified "split" and "image_mode" are inside the "root" directory')

    def _load_json(self, path):
        with open(path, 'r') as file:
            data = json.load(file)
        return data

class Cityscapes__(ConcatVisionDataset, CityscapesInfo):
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
                
        datasets = [VisionDataset(root, os.path.join(self.images_dir, city), image_type, convert, image_transforms) 
                    for city in sorted(glob.glob(os.path.join(root, self.images_dir, '*')))]
        super(Cityscapes__, self).__init__(datasets)
        
    def extra_repr(self):
        lines = ["Split: {split}", "Mode: {image_mode}", "Type: {image_type}"]
        return '\n'.join(lines).format(**self.__dict__)
    
    
class Cityscapes_(StackVisionDataset, CityscapesInfo):    
    Cityscapes__ = Cityscapes__
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
        
        datasets = [self.Cityscapes__(root, split, image_mode, image_type, image_transforms)
                    for image_type, image_transforms in zip(self.image_types, self.image_transforms)] 
        super(Cityscapes_, self).__init__(datasets)
        
    def extra_repr(self):
        lines = ["Split: {split}", "Mode: {image_mode}", "Types: {image_types}"]
        return '\n'.join(lines).format(**self.__dict__)

    @property
    def images(self):
        return list(zip(*[dataset.images for dataset in self.datasets]))

    
class Cityscapes(StackVisionDataset, CityscapesInfo):
    Cityscapes_ = Cityscapes_
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
            
        datasets = [self.Cityscapes_(root, split, _mode, _types, _transforms)
                    for _mode, _types, _transforms in zip(image_modes, image_types, self.image_transforms)]
        super(Cityscapes, self).__init__(datasets)
    
    def extra_repr(self):
        lines = ["Split: {split}", "Modes: {image_modes}", "Types: {image_types}"]
        return '\n'.join(lines).format(**self.__dict__)

    @property
    def images(self):
        return list(zip(*[dataset.images for dataset in self.datasets]))
