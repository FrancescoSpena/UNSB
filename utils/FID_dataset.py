# FID from official "https://github.com/mseitzer/pytorch-fid/blob/master/src/pytorch_fid/fid_score.py", with a slight modification due to imaginary components of Covariance Matrix 

from PIL import Image
import torch 
import numpy as np
from tqdm import tqdm
from torch.nn.functional import adaptive_avg_pool2d
import torchvision.transforms as TF
from scipy import linalg
import pathlib
import os 
from inception_v3 import InceptionV3

# Supported image formats
IMAGE_EXTENSIONS = {"bmp", "jpg", "jpeg", "pgm", "png", "ppm", "tif", "tiff", "webp"}  

class ImagePathDataset(torch.utils.data.Dataset):
    '''ImagePathDataset is a class for handle the images'''
    def __init__(self, files, transforms=None):
        self.files = files  # List of image file paths
        self.transforms = transforms  # Optional transforms to apply to images

    def __len__(self):
        '''Return the number of images'''
        return len(self.files)  

    def __getitem__(self, i):
        '''Return the item at position i'''
        path = self.files[i]  # Get the image path
        img = Image.open(path).convert("RGB")  # Open image and convert to RGB
        if self.transforms is not None:
            img = self.transforms(img)  # Apply transforms if any
        return img  

def get_activations(files, model, batch_size=50, dims=2048, device="cpu", num_workers=1):
    model.eval()  # Set model to evaluation mode

    if batch_size > len(files):
        print("Warning: batch size is bigger than the data size. Setting batch size to data size")
        batch_size = len(files)  # Adjust batch size if it's larger than the number of files

    dataset = ImagePathDataset(files, transforms=TF.ToTensor())  # Create dataset from image files
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )  # Create dataloader for batching

    pred_arr = np.empty((len(files), dims))  # Initialize array to hold activations

    start_idx = 0  # Start index for filling pred_arr

    for batch in tqdm(dataloader):
        batch = batch.to(device)  # Move batch to device

        with torch.no_grad():
            pred = model(batch)[0]  # Get model predictions

        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))  # Apply adaptive average pooling

        pred = pred.squeeze(3).squeeze(2).cpu().numpy()  # Convert predictions to numpy array

        pred_arr[start_idx : start_idx + pred.shape[0]] = pred  # Store predictions in pred_arr

        start_idx = start_idx + pred.shape[0]  # Update start index

    return pred_arr  # Return the array of activations

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    '''Compute FID value'''
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    # Ensure that sigma1 and sigma2 have the same shape
    max_shape = max(sigma1.shape[0], sigma2.shape[0])
    sigma1 = np.pad(sigma1, ((0, max_shape - sigma1.shape[0]), (0, max_shape - sigma1.shape[1])), mode='constant')
    sigma2 = np.pad(sigma2, ((0, max_shape - sigma2.shape[0]), (0, max_shape - sigma2.shape[1])), mode='constant')

    assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # Add a small value to the diagonal to improve numerical stability
    offset1 = np.eye(sigma1.shape[0]) * eps
    offset2 = np.eye(sigma2.shape[0]) * eps

    covmean, _ = linalg.sqrtm((sigma1 + offset1).dot(sigma2 + offset2), disp=False)
    if not np.isfinite(covmean).all():
        msg = "fid calculation produces singular product; adding %s to diagonal of cov estimates" % eps
        print(msg)
        covmean = linalg.sqrtm((sigma1 + offset1).dot(sigma2 + offset2))

    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

def calculate_activation_statistics(files, model, batch_size=50, dims=2048, device="cpu", num_workers=1):
    act = get_activations(files, model, batch_size, dims, device, num_workers)  # Get activations for the files
    mu = np.mean(act, axis=0)  # Calculate mean of activations
    sigma = np.cov(act, rowvar=False)  # Calculate covariance of activations
    return mu, sigma  

def compute_statistics_of_path(path, model, batch_size, dims, device, num_workers=1):
    if path.endswith(".npz"):
        with np.load(path) as f:
            m, s = f["mu"][:], f["sigma"][:]  # Load precomputed statistics
    else:
        path = pathlib.Path(path)
        files = sorted([file for ext in IMAGE_EXTENSIONS for file in path.glob("*.{}".format(ext))])  # Gather image files
        m, s = calculate_activation_statistics(files, model, batch_size, dims, device, num_workers)  # Compute statistics

    return m, s  # Return mean and covariance

def calculate_fid_given_paths(paths, batch_size, device, dims, num_workers=1):
    for p in paths:
        if not os.path.exists(p):
            raise RuntimeError("Invalid path: %s" % p)  # Check if paths exist

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]  # Get block index for InceptionV3

    model = InceptionV3([block_idx]).to(device)  # Initialize InceptionV3 model

    m1, s1 = compute_statistics_of_path(paths[0], model, batch_size, dims, device, num_workers)  # Compute statistics for first path
    m2, s2 = compute_statistics_of_path(paths[1], model, batch_size, dims, device, num_workers)  # Compute statistics for second path
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)  # Calculate FID

    return fid_value  # Return FID value

def save_fid_stats(paths, batch_size, device, dims, num_workers=1):
    if not os.path.exists(paths[0]):
        raise RuntimeError("Invalid path: %s" % paths[0])  # Check if input path exists

    if os.path.exists(paths[1]):
        raise RuntimeError("Existing output file: %s" % paths[1])  # Check if output file already exists

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]  # Get block index for InceptionV3

    model = InceptionV3([block_idx]).to(device)  # Initialize InceptionV3 model

    print(f"Saving statistics for {paths[0]}")

    m1, s1 = compute_statistics_of_path(paths[0], model, batch_size, dims, device, num_workers)  # Compute statistics for input path

    np.savez_compressed(paths[1], mu=m1, sigma=s1)  # Save statistics to file