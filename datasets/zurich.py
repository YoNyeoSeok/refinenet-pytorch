import os
from collections import namedtuple
import glob

from torchvision.datasets.utils import extract_archive, verify_str_arg, iterable_to_str
from torch.utils.data import Subset
from .vision import VisionDataset, ConcatVisionDataset, StackVisionDataset
from PIL import Image

class ZurichInfo:
    _repr_indent = VisionDataset._repr_indent
    
    ZurichClass = namedtuple('ZurichClass', ['name', 'id', 'train_id', 'category', 'category_id',
                                                     'has_instances', 'ignore_in_eval', 'color'])

    classes = [
        ZurichClass('unlabeled', 0, 255, 'void', 0, False, True, (0, 0, 0)),
        ZurichClass('ego vehicle', 1, 255, 'void', 0, False, True, (0, 0, 0)),
        ZurichClass('rectification border', 2, 255, 'void', 0, False, True, (0, 0, 0)),
        ZurichClass('out of roi', 3, 255, 'void', 0, False, True, (0, 0, 0)),
        ZurichClass('static', 4, 255, 'void', 0, False, True, (0, 0, 0)),
        ZurichClass('dynamic', 5, 255, 'void', 0, False, True, (111, 74, 0)),
        ZurichClass('ground', 6, 255, 'void', 0, False, True, (81, 0, 81)),
        ZurichClass('road', 7, 0, 'flat', 1, False, False, (128, 64, 128)),
        ZurichClass('sidewalk', 8, 1, 'flat', 1, False, False, (244, 35, 232)),
        ZurichClass('parking', 9, 255, 'flat', 1, False, True, (250, 170, 160)),
        ZurichClass('rail track', 10, 255, 'flat', 1, False, True, (230, 150, 140)),
        ZurichClass('building', 11, 2, 'construction', 2, False, False, (70, 70, 70)),
        ZurichClass('wall', 12, 3, 'construction', 2, False, False, (102, 102, 156)),
        ZurichClass('fence', 13, 4, 'construction', 2, False, False, (190, 153, 153)),
        ZurichClass('guard rail', 14, 255, 'construction', 2, False, True, (180, 165, 180)),
        ZurichClass('bridge', 15, 255, 'construction', 2, False, True, (150, 100, 100)),
        ZurichClass('tunnel', 16, 255, 'construction', 2, False, True, (150, 120, 90)),
        ZurichClass('pole', 17, 5, 'object', 3, False, False, (153, 153, 153)),
        ZurichClass('polegroup', 18, 255, 'object', 3, False, True, (153, 153, 153)),
        ZurichClass('traffic light', 19, 6, 'object', 3, False, False, (250, 170, 30)),
        ZurichClass('traffic sign', 20, 7, 'object', 3, False, False, (220, 220, 0)),
        ZurichClass('vegetation', 21, 8, 'nature', 4, False, False, (107, 142, 35)),
        ZurichClass('terrain', 22, 9, 'nature', 4, False, False, (152, 251, 152)),
        ZurichClass('sky', 23, 10, 'sky', 5, False, False, (70, 130, 180)),
        ZurichClass('person', 24, 11, 'human', 6, True, False, (220, 20, 60)),
        ZurichClass('rider', 25, 12, 'human', 6, True, False, (255, 0, 0)),
        ZurichClass('car', 26, 13, 'vehicle', 7, True, False, (0, 0, 142)),
        ZurichClass('truck', 27, 14, 'vehicle', 7, True, False, (0, 0, 70)),
        ZurichClass('bus', 28, 15, 'vehicle', 7, True, False, (0, 60, 100)),
        ZurichClass('caravan', 29, 255, 'vehicle', 7, True, True, (0, 0, 90)),
        ZurichClass('trailer', 30, 255, 'vehicle', 7, True, True, (0, 0, 110)),
        ZurichClass('train', 31, 16, 'vehicle', 7, True, False, (0, 80, 100)),
        ZurichClass('motorcycle', 32, 17, 'vehicle', 7, True, False, (0, 0, 230)),
        ZurichClass('bicycle', 33, 18, 'vehicle', 7, True, False, (119, 11, 32)),
        ZurichClass('license plate', -1, -1, 'vehicle', 7, False, True, (0, 0, 142)),
    ]

    def verify_mode(self, image_mode):
        valid_modes = ("gt_color", "color",
                       "gt_labelIds", "semantic",
                       "gt_labelTrainIds", "semantic_train",
                       "RGB", "RGB",)
        for i in range(len(valid_modes)//2):
            if image_mode == valid_modes[i*2+1]:
                image_mode = valid_modes[i*2]
                break
        verify_str_arg(image_mode, "image_mode", valid_modes)
        return image_mode
        
    def verify_mode_split(self, split, image_mode):
        image_mode = self.verify_mode(image_mode)
        
        if image_mode == "gt_color":
            valid_splits = ("testv1", "testv2")
        elif image_mode == "gt_labelIds":
            valid_splits = ("testv1", "testv2")
        elif image_mode == "gt_labelTrainIds":
            valid_splits = ("testv1", "testv2")
        elif image_mode == "RGB":
            valid_splits = ("light", "medium", "testv1", "testv2")
        
        msg = ("Unknown value '{}' for argument split if image_mode is '{}'. "
               "Valid values are {{{}}}.")
        msg = msg.format(split, image_mode, iterable_to_str(valid_splits))
        verify_str_arg(split, "split", valid_splits, msg)
        
        return image_mode
    
    # def verify_dataset(self):
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

class Zurich__(ConcatVisionDataset, ZurichInfo):
    def __init__(self, root, split='testv2',
                 image_mode='RGB', image_transforms=None,
                ):
        self.root = root
        self.split = split

        image_mode = self.verify_mode_split(split, image_mode)
        self.image_mode = image_mode
        
        self.image_transforms = image_transforms
        
        self.images_dir = self.image_mode
        # self.verify_dataset()
        
        convert = 'L' if 'Ids' in self.image_mode else 'RGB'
        datasets = [VisionDataset(root, os.path.join(self.images_dir, camera), '.png', convert, image_transforms) 
                    for camera in sorted(glob.glob(os.path.join(root, self.images_dir, '*')))]
        super(Zurich__, self).__init__(datasets)

    def extra_repr(self):
        lines = ["Split: {split}", "Mode: {image_mode}"]
        return '\n'.join(lines).format(**self.__dict__)


class Zurich_(Subset, ZurichInfo):
    __repr__ = Zurich__.__repr__
    _format_transform_repr = Zurich__._format_transform_repr
    extra_repr = Zurich__.extra_repr

    Zurich__ = Zurich__
    def __init__(self, root, split='train', # cities
                 image_mode='RGB', image_transforms=None,
                 lists_file_names='lists_file_names',
                ):
        dataset = self.Zurich__(root, split, image_mode, image_transforms)
        self.root = dataset.root
        self.split = dataset.split
        self.image_mode = dataset.image_mode
        self.image_transforms = dataset.image_transforms

        self.lists_file_names = lists_file_names

        lists_file_name = os.path.join(lists_file_names, "RGB_{}_filenames.txt".format(split))

        if os.path.isfile(os.path.join(root, lists_file_name)):
            with open(os.path.join(root, lists_file_name)) as f:
                lines = list(map(str.strip, f.readlines()))           
            # print(lines, dataset.images) 
            indices = [a for a, img in enumerate(dataset.images) for line in sorted(lines) if line[3:] in img]
        else:
            raise RuntimeError('refined_filenames are not found or incomplete. Please make sure the required file'
                                'is inside the "root" directory')
        super(Zurich_, self).__init__(dataset, indices)
    
    @property
    def images(self):
        return [self.dataset.images[idx] for idx in self.indices]

    def __repr__(self):
        head = self.__class__.__name__
        body = ["Number of datapoints: {}".format(self.__len__())]
        if self.root is not None:
            body.append("Root location: {}".format(self.root))
        body += self.extra_repr().splitlines()
        if self.dataset.transform is not None:
            body += self._format_transform_repr(self.dataset.transform, "Transform: ")
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return '\n'.join(lines)


class Zurich(StackVisionDataset, ZurichInfo):
    Zurich_ = Zurich_
    def __init__(self, root, split='train', # cities
                 image_modes=['RGB'], image_transforms=None,
                ):
        self.root = root
        self.split = split

        image_mode = [self.verify_mode_split(split, image_mode) for image_mode in image_modes]
        self.image_modes = image_mode

        image_transforms = (image_transforms 
                            if isinstance(image_transforms, list)
                            else [image_transforms for _mode in image_modes])
        self.image_transforms = image_transforms
            
        datasets = [self.Zurich_(root, split, _mode, _transforms)
                    for _mode, _transforms in zip(image_modes, self.image_transforms)]
        super(Zurich, self).__init__(datasets)
    
    def extra_repr(self):
        lines = ["Split: {split}", "Modes: {image_modes}"]
        return '\n'.join(lines).format(**self.__dict__)

    @property
    def images(self):
        return list(zip(*[dataset.images for dataset in self.datasets]))
