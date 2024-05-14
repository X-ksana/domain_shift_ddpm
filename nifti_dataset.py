import os
import nibabel as nib
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

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

        # Convert the image data to a PIL image
        pil_image = Image.fromarray(image_data)

        # Apply transformations if specified
        if self.transform:
            pil_image = self.transform(pil_image)

        # Convert the PIL image to a torch tensor
        image_tensor = transforms.ToTensor()(pil_image)

        return image_tensor
