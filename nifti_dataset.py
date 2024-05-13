# from nifti_dataset import NiftiDataset

import os
import nibabel as nib
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class NiftiDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.image_files = [file for file in os.listdir(root_dir) if file.endswith('.nii.gz')]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.root_dir, self.image_files[idx])
        
        # Load the NIfTI image
        nifti_image = nib.load(image_path)
        image_data = nifti_image.get_fdata()

        # Apply transformations if specified
        if self.transform:
            image_data = self.transform(image_data)

        # Convert image to torch tensor
        image_tensor = torch.from_numpy(image_data).float()

        return image_tensor

