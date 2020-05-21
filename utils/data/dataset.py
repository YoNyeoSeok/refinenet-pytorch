from torch.utils import data

class ConcatDataset(data.ConcatDataset):
    def __init__(self, *args, **kwds):
        super(ConcatDataset, self).__init__(*args, **kwds)
        
    def __repr__(self):
        head = "ConcatDataset " + self.__class__.__name__
        lines = [head] + [repr(dataset) for dataset in self.datasets]
        return '\n'.join(lines)

class StackDataset(data.Dataset):
    r"""Dataset as a stack of multiple datasets.
    This class is useful to assemble different existing datasets.
    Arguments:
        datasets (sequence): List of datasets to be stacked
    """

    def __init__(self, datasets):
        super(StackDataset, self).__init__()
        assert len(datasets) > 0, 'datasets should not be an empty iterable'
        self.datasets = list(datasets)
        for i, d in enumerate(self.datasets):
            assert not isinstance(d, data.IterableDataset), "StackDataset does not support IterableDataset"
            if i == 0:
                l = len(d)
            else:
                assert l==len(d), "datasets should have same length"

    def __len__(self):
        return len(self.datasets[0])

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        return tuple([dataset[idx] for dataset in self.datasets])

    def __repr__(self):
        head = "StackDataset " + self.__class__.__name__
        lines = [head] + [repr(dataset) for dataset in self.datasets]
        return '\n'.join(lines)
