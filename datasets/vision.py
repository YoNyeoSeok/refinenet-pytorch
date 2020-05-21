import json
import os
from collections import namedtuple
import zipfile

from torchvision.datasets.utils import extract_archive, verify_str_arg, iterable_to_str
from torch.utils.data import Dataset
from utils.data import ConcatDataset, StackDataset
from PIL import Image


class VisionDataset(Dataset):
    """
    Args:
        root (string): Root directory of dataset.
        split (string): Dataset split. E.g, ``train``
        mode (string): Dataset mode. E.g, ``gtFine``
        substring (string): image substring. E.g, ``.png``
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
    """
    _repr_indent = 4

    def __init__(self, root, images_dir, substring='.png', convert='RGB', transform=None,
                ):
        super(Dataset, self).__init__()
        
        self.root = root
        self.images_dir = images_dir
        self.substring = substring
        self.convert = convert
        self.transform = transform
        
        self.images = []

        if not os.path.isdir(os.path.join(self.root, self.images_dir)):
            raise RuntimeError('Dataset not found or incomplete. Please make sure'
                               'the "images_dir" directory is exist')

        self.images = sorted([os.path.join(self.root, self.images_dir, file_name)
                              for file_name in os.listdir(os.path.join(self.root, self.images_dir))
                              if substring in file_name])

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tensor: image
        """
        
        images = []
        image = Image.open(self.images[index]).convert(self.convert)
        if self.transform is not None:
            image = self.transform(image)

        return image

    def __len__(self):
        return len(self.images)

    def __repr__(self):
        head = self.__class__.__name__
        body = ["Number of datapoints: {}".format(self.__len__())]
        if self.root is not None:
            body.append("Root location: {}".format(self.root))
        body += self.extra_repr().splitlines()
        if self.transform is not None:
            body += self._format_transform_repr(self.transform, "Transform: ")
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return '\n'.join(lines)

    def _format_transform_repr(self, transform, head):
        lines = transform.__repr__().splitlines()
        return (["{}{}".format(head, lines[0])] +
                ["{}{}".format(" " * len(head), line) for line in lines[1:]])

    def extra_repr(self):
        return ""

    
class ConcatVisionDataset(ConcatDataset):
    _repr_indent = VisionDataset._repr_indent
    __repr__ = VisionDataset.__repr__
    _format_transform_repr = VisionDataset._format_transform_repr
    extra_repr = VisionDataset.extra_repr

    def __init__(self, *args, **kwds):
        super(ConcatVisionDataset, self).__init__(*args, **kwds)
        self.root = self.datasets[0].root
        self.substring = self.datasets[0].substring
        self.convert = self.datasets[0].convert
        self.transform = self.datasets[0].transform
        # self.extra_repr = self.datasets[0].extra_repr

        for i, d in enumerate(self.datasets):
            assert self.root == d.root, "datasets should have same root"
            assert self.substring==d.substring, "datasets should have same substring"
            assert self.convert==d.convert, "datasets should have same convert"
            assert self.transform==d.transform, "datasets should have same transform"
            # assert self.extra_repr()==d.extra_repr(), "datasets should have same extra_repr"
        
    @property
    def images(self):
        images = self.datasets[0].images
        for dataset in self.datasets[1:]:
            images += dataset.images
        return images
    
        
class StackVisionDataset(StackDataset):
    _repr_indent = VisionDataset._repr_indent
    _format_transform_repr = VisionDataset._format_transform_repr
    extra_repr = VisionDataset.extra_repr

    def __init__(self, *args, **kwds):
        super(StackVisionDataset, self).__init__(*args, **kwds)
        self.root = self.datasets[0].root
        # self.extra_repr = self.datasets[0].extra_repr

        for i, d in enumerate(self.datasets):
            assert self.root == d.root, "datasets should have same root"
            # assert self.extra_repr()==d.extra_repr(), "datasets should have same extra_repr"
    
    @property
    def images(self):
        return list(zip(*[dataset.images for dataset in self.datasets]))
    
    def __repr__(self):
        head = self.__class__.__name__
        body = ["Number of datapoints: {}".format(self.__len__())]
        if self.root is not None:
            body.append("Root location: {}".format(self.root))
        body += self.extra_repr().splitlines()
        def transform_body(d, head):
            if hasattr(d, 'transform'):
            # if isinstance(d, (VisionDataset, ConcatVisionDataset)):
                return self._format_transform_repr(d.transform, head)
            elif hasattr(d, 'datasets'):
            # elif isinstance(d, StackVisionDataset):
                return [head] + sum([transform_body(d_, "{}({}): ".format(" " * len(head), i_))
                                     for i_, d_ in enumerate(d.datasets)],
                                    [])
            elif hasattr(d, 'dataset'):
            # if isinstance(d, SubsetVision):
                return transform_body(d.dataset, head)
            else:
                return []
        body += transform_body(self, "Transform:")
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return '\n'.join(lines)
