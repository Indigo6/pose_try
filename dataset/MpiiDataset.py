import torch.utils.data as data


class MpiiDataset(data.Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None):
        self.sample_set = sample_set
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        sample = sample_set[index]
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.data)