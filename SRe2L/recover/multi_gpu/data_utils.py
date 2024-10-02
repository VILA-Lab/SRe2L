
# ------------------------------------------------------------------------------
# Multi-GPU training code for recover in SRe2L is modified from https://github.com/VILA-Lab/SRe2L 
# This version is revised by Xiaochen Ma (https://ma.xiaochen.world/)
# ------------------------------------------------------------------------------

import os
import torch
from PIL import Image
import torch.distributed as dist
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, DistributedSampler

class MultiCardDataset(Dataset):
    def __init__(self, num_classes, ipc, root_dir='./syn_data'):
        """
        Initialize the dataset.

        Parameters:
            num_classes (int): Number of classes.
            ipc (int): Number of instances per class.
            root_dir (str): Path to the directory containing images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.num_classes = num_classes
        self.ipc = ipc
        self.root_dir = root_dir
        self.total_samples = self.num_classes * self.ipc

    def __len__(self):
        """
        Return the total number of samples in the dataset.
        """
        return self.total_samples

    def __getitem__(self, idx):
        """
        Retrieve a sample by its index.

        Parameters:
            idx (int): Index of the sample.

        Returns:
            dict: A dictionary containing 'class_id', 'path', and 'image'.
        """
        if idx < 0 or idx >= self.total_samples:
            raise IndexError("Index out of range")

        # Calculate class ID and IPC ID
        class_id = idx // self.ipc  # Integer division for class ID
        ipc_id = idx % self.ipc      # Modulo for IPC ID

        # Generate the file path
        path = f"{self.root_dir}/new{class_id:03d}/class{class_id:03d}_id{ipc_id:03d}.jpg"

        return {
            'class_id': class_id,
            'path': path,
        }

"""
Code Below is for testing the dataset on multiple GPUs.
"""

def setup(rank, world_size):
    """
    Initialize the distributed environment.
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    """
    Clean up the distributed environment.
    """
    dist.destroy_process_group()

def worker(rank, world_size, num_classes, ipc):
    """
    Worker function to load and test the dataset on each GPU.
    """
    setup(rank, world_size)

    # Create dataset and data sampler
    dataset = MultiCardDataset(num_classes=num_classes, ipc=ipc)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
    dataloader = DataLoader(dataset, batch_size=4, sampler=sampler, shuffle=False)

    # Iterate over the data
    for batch in dataloader:
        print(f"Rank {rank} - Batch: {batch['class_id']} - Paths: {batch['path']}")

    cleanup()

if __name__ == "__main__":
    import torch.multiprocessing as mp
    num_classes = 5  # Number of classes
    ipc = 2         # Instances per class
    world_size = 4   # Number of GPUs to simulate

    # Start the multiprocessing
    mp.spawn(worker, args=(world_size, num_classes, ipc), nprocs=world_size, join=True)