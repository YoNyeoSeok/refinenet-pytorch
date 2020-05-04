import os

from torchvision.datasets.utils import verify_str_arg
from torchvision.datasets.vision import VisionDataset
from PIL import Image

class Zurich(VisionDataset):
    
    def __init__(self, root, split, target_type, *args, **kwds):
        super(Zurich, self).__init__(root, *args, **kwds)
        self.split = split
        self.target_type = target_type

        verify_str_arg(split, "split", ('testv1', 'testv2', 'light', 'medium'))
        if split in ['testv2', 'testv1']:
            valid_target_type = ('gt_labelIds', 'gt_labelTrainIds', 'RGB', 'none')
        else:
            valid_target_type = ('none')
            
        if not isinstance(target_type, list):
            self.target_type = [target_type]
        [verify_str_arg(value, "target_type", valid_target_type)
         for value in self.target_type]
        
        image_file = os.path.join(root, 'lists_file_names', 'RGB_{}_filenames.txt'.format(split))
        if os.path.isfile(image_file):
            with open(image_file, 'r') as f:
                self.images = [os.path.join(root, line.strip()) for line in f.readlines()]
        else:
            raise RuntimeError('Image file not found in {}.'.format(image_file))
            
        self.targets = []
        for img in self.images:
            target_types = []
            for t in self.target_type:
                if t != 'none':
                    target_types.append(img.replace('RGB', t))
            self.targets.append(target_types)
            
    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert('RGB')
        
        targets = []
        targets = [Image.open(self.targets[index][i]) for i, t in enumerate(self.target_type) if t != 'none']
        target = tuple(targets) if len(targets) > 1 else targets[0]

        if self.transforms is not None:
            image, target = self.transforms(image, target)
            
        return image, target

    def __len__(self):
        return len(self.images)
    
    def extra_repr(self):
        lines = ["Environment: foggy", "Split: {}".format(self.split), "Type: {}".format(self.target_type)]
        return '\n'.join(lines).format(**self.__dict__)
    