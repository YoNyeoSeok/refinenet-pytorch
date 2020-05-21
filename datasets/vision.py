import json
import os
from collections import namedtuple
import zipfile

from torchvision.datasets.utils import extract_archive, verify_str_arg, iterable_to_str
from torch.utils.data import Dataset
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
        head = "Dataset " + self.__class__.__name__
        body = ["Number of datapoints: {}".format(self.__len__())]
        if self.root is not None:
            body.append("Root location: {}".format(self.root))
        body += self.extra_repr().splitlines()
        if hasattr(self, "transforms") and self.transforms is not None:
            body += [repr(self.transforms)]
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return '\n'.join(lines)

    def _format_transform_repr(self, transform, head):
        lines = transform.__repr__().splitlines()
        return (["{}{}".format(head, lines[0])] +
                ["{}{}".format(" " * len(head), line) for line in lines[1:]])

    def extra_repr(self):
        return ""