""" Defining Dataset """

import os
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class ImageDataset(Dataset):
    def __init__(self, img_dir):
        self.img_dir = img_dir  # Store the directory path to images
        self.image_paths = []  # List to hold the paths of the images
        
        # Walk through the directory to collect all image file paths
        for root, dirs, files in os.walk(img_dir):
            for file in files:
                # Add paths for files with appropriate image extensions
                if file.endswith(('.png', '.jpg', '.jpeg')):  
                    self.image_paths.append(os.path.join(root, file))

        # Define transformations for image preprocessing
        self.transform = transforms.Compose([
            transforms.ToPILImage(),  # Convert numpy arrays to PIL images for further processing
            transforms.Resize((256, 256)),  # Resize images to uniform size for the model
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Normalize pixel values to [-1, 1]
        ])

    def __len__(self):
        # Return the total number of images available in the dataset
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Fetch the path and load the image
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert color from BGR to RGB

        # Apply the predefined transformations to the image
        if self.transform:
            image = self.transform(image)
        
        return image  # Return the transformed image