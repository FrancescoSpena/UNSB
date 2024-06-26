""" Computing FID per epoch """

import numpy as np
import torch 
from torch.nn.functional import adaptive_avg_pool2d
from scipy.linalg import sqrtm

def epoch_calculate_activation_statistics(images, model, batch_size=128, dims=2048, cuda=False):
    # Set the model to evaluation mode
    model.eval()  
    
    # Select device
    if cuda:
        model = model.cuda()  
        images = images.cuda()
    else:
        model = model.cpu()  
        images = images.cpu()  

    act = np.empty((len(images), dims))
    
    # No need to track gradients for this operation
    with torch.no_grad():  
        pred = model(images)
        pred = pred[0]
        
        # Check if the output is 4D (batch, channels, height, width)
        if pred.dim() == 4:  
            if pred.size(2) != 1 or pred.size(3) != 1:
                pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
            pred = pred.view(pred.size(0), -1)
        
        # Check if the output is 2D (batch, features)
        elif pred.dim() == 2:  
            pred = pred
        else:
            raise RuntimeError("Unexpected output dimensions from the model.")

        act = pred.cpu().numpy()

    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma

def epoch_calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('FID calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = sqrtm((sigma1 + offset).dot(sigma2 + offset))
   
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


def epoch_calculate_fretchet(images_real, images_fake, model):
    '''Calculate final value '''
    mu_1, std_1 = epoch_calculate_activation_statistics(images_real, model, cuda=True)   
    mu_2, std_2 = epoch_calculate_activation_statistics(images_fake, model, cuda=True)
    
    # Get Frechet distance
    fid_value = epoch_calculate_frechet_distance(mu_1, std_1, mu_2, std_2)
    return fid_value