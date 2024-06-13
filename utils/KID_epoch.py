""" Computing KID per epoch """ 

from sklearn.metrics.pairwise import polynomial_kernel
import numpy as np 
from tqdm import tqdm 
import torch 
from torch.nn.functional import adaptive_avg_pool2d

def epoch_polynomial_mmd(codes_g, codes_r, degree=3, gamma=None, coef0=1, var_at_m=None, ret_var=True):
    '''Compute the polynomial MMD (Maximum Mean Discrepancy)'''
    # Compute the polynomial kernel for generated vs generated, real vs real, and generated vs real
    K_XX = polynomial_kernel(codes_g, codes_g, degree=degree, gamma=gamma, coef0=coef0)
    K_YY = polynomial_kernel(codes_r, codes_r, degree=degree, gamma=gamma, coef0=coef0)
    K_XY = polynomial_kernel(codes_g, codes_r, degree=degree, gamma=gamma, coef0=coef0)

    # Calculate the MMD value
    m = K_XX.shape[0]
    mmd2 = (np.mean(K_XX) + np.mean(K_YY) - 2 * np.mean(K_XY))

    if not ret_var:
        return mmd2

    return mmd2, None

def epoch_polynomial_mmd_averages(codes_g, codes_r, n_subsets=10, subset_size=1000, ret_var=True, output=sys.stdout, **kernel_args):
    '''Compute the polynomial MMD averages over multiple subsets'''
    # Adjust subset size if it's larger than the number of available codes
    actual_subset_size = min(subset_size, len(codes_g), len(codes_r))

    m = min(len(codes_g), len(codes_r))
    mmds = np.zeros(n_subsets)
    vars = np.zeros(n_subsets) if ret_var else None
    choice = np.random.choice

    # Iterate over the number of subsets and compute MMD for each subset
    with tqdm(range(n_subsets), desc='MMD', file=output) as bar:
        for i in bar:
            if actual_subset_size < subset_size:
                # If the actual subset size is smaller than desired, allow replacement
                g = codes_g[choice(len(codes_g), actual_subset_size, replace=True)]
                r = codes_r[choice(len(codes_r), actual_subset_size, replace=True)]
            else:
                g = codes_g[choice(len(codes_g), actual_subset_size, replace=False)]
                r = codes_r[choice(len(codes_r), actual_subset_size, replace=False)]
            o = epoch_polynomial_mmd(g, r, **kernel_args, var_at_m=m, ret_var=ret_var)
            if ret_var:
                mmds[i], vars[i] = o
            else:
                mmds[i] = o
            bar.set_postfix({'mean': mmds[:i+1].mean()})
    return (mmds, vars) if ret_var else mmds

def epoch_calculate_kid_given_activations(activations_real, activations_fake):
    '''Compute KID (Kernel Inception Distance) given activations'''
    return epoch_polynomial_mmd_averages(activations_real, activations_fake, n_subsets=10)

def epoch_calculate_activations(images, model, cuda=False):
    '''Compute activations of images using the model'''
    model.eval()
    batch_size = images.size(0)
    if cuda:
        images = images.cuda()
        model.cuda()
    with torch.no_grad():
        pred = model(images)
        pred = pred[0]
        # Check if the output is 4D (batch, channels, height, width)
        if pred.dim() == 4:  
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
        pred = pred.view(batch_size, -1)
    # Return as numpy array for consistency with KID function
    return pred.cpu().numpy()  

def epoch_compute_mmd_simple(codes_g, codes_r, degree=3, gamma=None, coef0=1):
    '''Compute MMD (Maximum Mean Discrepancy) using a simple polynomial kernel'''
    if gamma is None:
        # Default gamma is 1/number of features
        gamma = 1.0 / codes_g.shape[1]  

    # Compute the polynomial kernel for generated vs generated, real vs real, and generated vs real
    K_XX = polynomial_kernel(codes_g, codes_g, degree=degree, gamma=gamma, coef0=coef0)
    K_YY = polynomial_kernel(codes_r, codes_r, degree=degree, gamma=gamma, coef0=coef0)
    K_XY = polynomial_kernel(codes_g, codes_r, degree=degree, gamma=gamma, coef0=coef0)

    # Calculate the MMD value
    mmd2 = np.mean(K_XX) + np.mean(K_YY) - 2 * np.mean(K_XY)
    return mmd2