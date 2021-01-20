import bisect
import random
import warnings

from torch._utils import _accumulate
from torch import randperm
# No 'default_generator' in torch/__init__.pyi
from torch import Tensor, Generator
from torch.utils.data import Dataset, Subset


def pseudo_random_split(dataset, lengths, seed=42):
    r"""
    Randomly split a dataset into non-overlapping new datasets of given lengths.
    Optionally fix the generator for reproducible results, e.g.:
    >>> random_split(range(10), [3, 7], generator=torch.Generator().manual_seed(42))
    Arguments:
        dataset (Dataset): Dataset to be split
        lengths (sequence): lengths of splits to be produced
        generator (Generator): Generator used for the random permutation.
    """
    generator = Generator().manual_seed(seed)
    # Cannot verify that dataset is Sized
    if sum(lengths) != len(dataset):  # type: ignore
        raise ValueError(
            "Sum of input lengths does not equal the length of the input dataset!")

    indices = randperm(sum(lengths), generator=generator).tolist()
    return [Subset(dataset, indices[offset - length: offset]) for offset, length in zip(_accumulate(lengths), lengths)]


def get_partial_dataset(dataset, percent, seed=42):
    if percent < 1:
        datasetLen = len(dataset)
        selectionLen = int(datasetLen*percent)
        if selectionLen < 1:  # Keep one sample at least
            selectionLen = 1
        return pseudo_random_split(dataset, [selectionLen, datasetLen-selectionLen])[0]
    else:
        return dataset


def split_dataset(dataset, percent, seed=42):
    if percent < 1:
        datasetLen = len(dataset)
        selectionLen = int(datasetLen*percent)
        if selectionLen < 1:  # Keep one sample at least
            selectionLen = 1
        return pseudo_random_split(dataset, [selectionLen, datasetLen-selectionLen])
    else:
        return [dataset, []]


def split_dataset_by_list(dataset, fileNameList):
    selectedIndices = []
    unselectedIndices = []
    fileNameSet = set(fileNameList)
    for index, l in enumerate(dataset.list):
        l_info = l.split()
        img_name, _ = l_info[0], l_info[1]
        if img_name in fileNameSet:
            fileNameSet.remove(img_name)
            selectedIndices.append(index)
        else:
            unselectedIndices.append(index)
    if len(fileNameSet) > 0:
        print('Exist missing sample in extra list: %s' % str(fileNameSet))
    return [Subset(dataset, selectedIndices), Subset(dataset, unselectedIndices)]


class ToyDataset(Dataset):
    def __init__(self, len):
        self.list = [str(num)+' '+str(num) for num in range(len)]

    def __getitem__(self, idx):
        return self.list[idx]

    def __len__(self):
        return len(self.list)


if __name__ == "__main__":
    # print(list(split_dataset(range(20), 0.0001)))
    # print(list(split_dataset(range(20), 0.1)))
    # print(list(split_dataset(range(20), 1)))
    toyDs = ToyDataset(20)
    dsA, dsB = split_dataset_by_list(toyDs, ['3', '5', '19', '33'])
    print(list(dsA))
    print(list(dsB))
