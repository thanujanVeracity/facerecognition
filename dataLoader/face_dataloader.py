import random
import torch
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler


from torch.utils.data import Dataset
from dataLoader.face_dataset import TripletFaceDatset
from torch.utils.data import DataLoader, random_split


class FaceDataLoader(DataLoader):
    
    def __init__(self, dataset: Dataset, batchsize: int, train_ratio: int, valid_ratio:int, transform):
        
        
        self.batch_size = batchsize
        self.dataset = dataset
        
        
        # Shuffle the dataset       
        test_ratio = 1- (train_ratio + valid_ratio)
        
        # Split your dataset into training, validation, and test sets
        train_indices, val_indices, test_indices = self._split_dataset(train_ratio, valid_ratio, test_ratio, shuffle=True, random_seed=42)
        
        # Create DataLoaders for each split
        train_loader, val_loader, test_loader = self._create_data_loaders( train_indices, val_indices, test_indices, batch_size=32)
        
        
        # set the loaders
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
    def get_train_loader(self):
        return self.train_loader

    def get_valid_loader(self):
        return self.val_loader

    def get_test_loader(self):
        return self.test_loader
    
    
    def _split_dataset(self, train_size, val_size, test_size, shuffle=True, random_seed=42):
        """
        Split a dataset into train, validation, and test sets.
        
        Args:
            dataset (torch.utils.data.Dataset): The dataset to split.
            train_size (float): Proportion of the dataset to use for training.
            val_size (float): Proportion of the dataset to use for validation.
            test_size (float): Proportion of the dataset to use for testing.
            shuffle (bool, optional): Whether to shuffle the dataset before splitting. Default is True.
            random_seed (int, optional): Random seed to use for shuffling. Default is 42.
        
        Returns:
            (train_indices, val_indices, test_indices): Tuple of lists of indices for the train, validation, and test sets.
        """
        assert train_size + val_size + test_size == 1, "train_size, val_size, and test_size should add up to 1."
        num_examples = len(self.dataset)
        indices = list(range(num_examples))
        if shuffle:
            torch.manual_seed(random_seed)
            random.shuffle(indices)
        train_cutoff = int(num_examples * train_size)
        val_cutoff = int(num_examples * (train_size + val_size))
        train_indices = indices[:train_cutoff]
        val_indices = indices[train_cutoff:val_cutoff]
        test_indices = indices[val_cutoff:]
        return train_indices, val_indices, test_indices

    def _create_data_loaders(self, train_indices, val_indices, test_indices, batch_size=32):
        """
        Create PyTorch data loaders for a dataset based on train, validation, and test set indices.
        
        Args:
            dataset (torch.utils.data.Dataset): The dataset to create data loaders for.
            train_indices (list): List of indices for the train set.
            val_indices (list): List of indices for the validation set.
            test_indices (list): List of indices for the test set.
            batch_size (int, optional): Batch size to use for data loaders. Default is 32.
            
        Returns:
            (train_loader, val_loader, test_loader): Tuple of PyTorch data loaders for the train, validation, and test sets.
        """
        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)
        test_sampler = SubsetRandomSampler(test_indices)
        train_loader = DataLoader(self.dataset, batch_size=batch_size, sampler=train_sampler)
        val_loader = DataLoader(self.dataset, batch_size=batch_size, sampler=val_sampler)
        test_loader = DataLoader(self.dataset, batch_size=batch_size, sampler=test_sampler)
        return train_loader, val_loader, test_loader
    
        
    
    
    
