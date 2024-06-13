"""Computing KID for the entire dataset """

import numpy as np 
from tqdm import tqdm 
from PIL import Image 
import torch 
from torch.nn.functional import adaptive_avg_pool2d
import os 
from inception_v3 import InceptionV3
from sklearn.metrics.pairwise import polynomial_kernel
import sys 

# Add import LeNet 

#KID for entire dataset
def get_activations(files, model, batch_size=1, dims=2048,
                    cuda=False, verbose=False):
    """Calculates the activations of the pool_3 layer for all images."""
    # Set the model to evaluation mode
    model.eval()
    # Determine if the input is numpy arrays
    is_numpy = True if type(files[0]) == np.ndarray else False

    # Check if the number of images is a multiple of the batch size
    if len(files) % batch_size != 0:
        print(('Warning: number of images is not a multiple of the '
               'batch size. Some samples are going to be ignored.'))
    # Adjust batch size if it is larger than the number of images
    if batch_size > len(files):
        print(('Warning: batch size is bigger than the data size. '
               'Setting batch size to data size'))
        batch_size = len(files)

    # Calculate the number of batches and the number of images used
    n_batches = len(files) // batch_size
    n_used_imgs = n_batches * batch_size

    # Initialize an array to store the activations
    pred_arr = np.empty((n_used_imgs, dims))

    # Loop through each batch
    for i in tqdm(range(n_batches)):
        if verbose:
            print('\rPropagating batch %d/%d' % (i + 1, n_batches), end='', flush=True)
        start = i * batch_size
        end = start + batch_size

        # Preprocess images based on their format
        if is_numpy:
            images = np.copy(files[start:end]) + 1
            images /= 2.
        else:
            images = [np.array(Image.open(str(f))) for f in files[start:end]]
            images = np.stack(images).astype(np.float32) / 255.
            images = torch.from_numpy(images)
            if len(images.shape) == 3:
                images = torch.unsqueeze(images, dim=-1).expand(-1, -1, -1, 3)
            images = images.permute((0, 3, 1, 2))

        batch = images.float()
        if cuda:
            batch = batch.cuda()

        # Get the model predictions
        pred = model(batch)[0]

        # Apply adaptive average pooling if the output dimensions are not 1x1
        if pred.shape[2] != 1 or pred.shape[3] != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        # Store the activations in the array
        pred_arr[start:end] = pred.cpu().data.numpy().reshape(batch_size, -1)

    if verbose:
        print('done', torch.min(images))

    return pred_arr

def extract_lenet_features(imgs, net):
    """Extract features using a LeNet model."""
    net.eval()
    feats = []
    imgs = imgs.reshape([-1, 100] + list(imgs.shape[1:]))
    if imgs[0].min() < -0.001:
        imgs = (imgs + 1) / 2.0
    print(imgs.shape, imgs.min(), imgs.max())
    imgs = torch.from_numpy(imgs)
    for i, images in enumerate(imgs):
        feats.append(net.extract_features(images).detach().cpu().numpy())
    feats = np.vstack(feats)
    return feats

def _compute_activations(path, model, batch_size, dims, cuda, model_type):
    """Compute activations for the given path using the specified model."""
    if not type(path) == np.ndarray:
        import glob
        jpg = os.path.join(path, '*.jpg')
        png = os.path.join(path, '*.png')
        path = glob.glob(jpg) + glob.glob(png)
        if len(path) > 50000:
            import random
            random.shuffle(path)
            path = path[:50000]
    if model_type == 'inception':
        act = get_activations(path, model, batch_size, dims, cuda)
    elif model_type == 'lenet':
        act = extract_lenet_features(path, model)
    return act

def calculate_kid_given_paths(paths, batch_size, cuda, dims, model_type='inception'):
    """Calculates the KID of two paths"""
    pths = []
    for p in paths:
        if not os.path.exists(p):
            raise RuntimeError('Invalid path: %s' % p)
        if os.path.isdir(p):
            pths.append(p)
        elif p.endswith('.npy'):
            np_imgs = np.load(p)
            if np_imgs.shape[0] > 50000: np_imgs = np_imgs[np.random.permutation(np.arange(np_imgs.shape[0]))][:50000]
            pths.append(np_imgs)

    if model_type == 'inception':
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        model = InceptionV3([block_idx])
    elif model_type == 'lenet':
        model = LeNet5()
        model.load_state_dict(torch.load('./models/lenet.pth'))
    if cuda:
        model.cuda()

    act_true = _compute_activations(pths[0], model, batch_size, dims, cuda, model_type)
    pths = pths[1:]
    results = []
    for j, pth in enumerate(pths):
        print(paths[j + 1])
        actj = _compute_activations(pth, model, batch_size, dims, cuda, model_type)
        kid_values = polynomial_mmd_averages(act_true, actj, n_subsets=100)
        results.append((paths[j + 1], kid_values[0].mean(), kid_values[0].std()))
    return results

def _sqn(arr):
    """Square norm of the flattened array."""
    flat = np.ravel(arr)
    return flat.dot(flat)

def polynomial_mmd_averages(codes_g, codes_r, n_subsets=50, subset_size=1000,
                            ret_var=True, output=sys.stdout, **kernel_args):
    """Compute MMD averages over several subsets."""
    m = min(codes_g.shape[0], codes_r.shape[0])
    subset_size = min(subset_size, m)  # Ensure subset_size is not larger than available samples
    mmds = np.zeros(n_subsets)
    if ret_var:
        vars = np.zeros(n_subsets)
    choice = np.random.choice

    with tqdm(range(n_subsets), desc='MMD', file=output) as bar:
        for i in bar:
            g = codes_g[choice(len(codes_g), subset_size, replace=False)]
            r = codes_r[choice(len(codes_r), subset_size, replace=False)]
            o = polynomial_mmd(g, r, **kernel_args, var_at_m=m, ret_var=ret_var)
            if ret_var:
                mmds[i], vars[i] = o
            else:
                mmds[i] = o
            bar.set_postfix({'mean': mmds[:i+1].mean()})
    return (mmds, vars) if ret_var else mmds

def polynomial_mmd(codes_g, codes_r, degree=3, gamma=None, coef0=1,
                   var_at_m=None, ret_var=True):
    """Compute polynomial MMD between two sets of codes."""
    X = codes_g
    Y = codes_r

    K_XX = polynomial_kernel(X, degree=degree, gamma=gamma, coef0=coef0)
    K_YY = polynomial_kernel(Y, degree=degree, gamma=gamma, coef0=coef0)
    K_XY = polynomial_kernel(X, Y, degree=degree, gamma=gamma, coef0=coef0)

    return _mmd2_and_variance(K_XX, K_XY, K_YY,
                              var_at_m=var_at_m, ret_var=ret_var)

def _mmd2_and_variance(K_XX, K_XY, K_YY, unit_diagonal=False,
                       mmd_est='unbiased', block_size=1024,
                       var_at_m=None, ret_var=True):
    """Calculate the MMD2 (Maximum Mean Discrepancy) and its variance."""
    
    m = K_XX.shape[0]  # Number of samples

    # Ensure that the kernel matrices have the correct shapes
    assert K_XX.shape == (m, m)
    assert K_XY.shape == (m, m)
    assert K_YY.shape == (m, m)

    # If var_at_m is not provided, set it to m
    if var_at_m is None:
        var_at_m = m

    # Initialize diagonal and sum variables based on whether unit diagonal is used
    if unit_diagonal:
        diag_X = diag_Y = 1
        sum_diag_X = sum_diag_Y = m
        sum_diag2_X = sum_diag2_Y = m
    else:
        diag_X = np.diagonal(K_XX)
        diag_Y = np.diagonal(K_YY)
        sum_diag_X = diag_X.sum()
        sum_diag_Y = diag_Y.sum()
        sum_diag2_X = _sqn(diag_X)
        sum_diag2_Y = _sqn(diag_Y)

    # Calculate the sum of the kernel matrices excluding the diagonal
    Kt_XX_sums = K_XX.sum(axis=1) - diag_X
    Kt_YY_sums = K_YY.sum(axis=1) - diag_Y
    K_XY_sums_0 = K_XY.sum(axis=0)
    K_XY_sums_1 = K_XY.sum(axis=1)

    Kt_XX_sum = Kt_XX_sums.sum()
    Kt_YY_sum = Kt_YY_sums.sum()
    K_XY_sum = K_XY_sums_0.sum()

    # Calculate MMD2 based on the chosen estimation method
    if mmd_est == 'biased':
        mmd2 = ((Kt_XX_sum + sum_diag_X) / (m * m)
                + (Kt_YY_sum + sum_diag_Y) / (m * m)
                - 2 * K_XY_sum / (m * m))
    else:
        assert mmd_est in {'unbiased', 'u-statistic'}
        mmd2 = (Kt_XX_sum + Kt_YY_sum) / (m * (m - 1))
        if mmd_est == 'unbiased':
            mmd2 -= 2 * K_XY_sum / (m * m)
        else:
            mmd2 -= 2 * (K_XY_sum - np.trace(K_XY)) / (m * (m - 1))

    # Return MMD2 if variance is not required
    if not ret_var:
        return mmd2

    # Calculate terms needed for variance estimation
    Kt_XX_2_sum = _sqn(K_XX) - sum_diag2_X
    Kt_YY_2_sum = _sqn(K_YY) - sum_diag2_Y
    K_XY_2_sum = _sqn(K_XY)
    dot_XX_XY = Kt_XX_sums.dot(K_XY_sums_1)
    dot_YY_YX = Kt_YY_sums.dot(K_XY_sums_0)
    
    if m <= 2:
        # Return zero variance if m is less than or equal to 2
        return mmd2, 0
    
    m1 = m - 1
    m2 = m - 2

    # Estimate zeta1 and zeta2 for variance calculation
    zeta1_est = (
        1 / (m * m1 * m2) * (
            _sqn(Kt_XX_sums) - Kt_XX_2_sum + _sqn(Kt_YY_sums) - Kt_YY_2_sum)
        - 1 / (m * m1) ** 2 * (Kt_XX_sum ** 2 + Kt_YY_sum ** 2)
        + 1 / (m * m * m1) * (
            _sqn(K_XY_sums_1) + _sqn(K_XY_sums_0) - 2 * K_XY_2_sum)
        - 2 / m ** 4 * K_XY_sum ** 2
        - 2 / (m * m * m1) * (dot_XX_XY + dot_YY_YX)
        + 2 / (m ** 3 * m1) * (Kt_XX_sum + Kt_YY_sum) * K_XY_sum
    )
    zeta2_est = (
        1 / (m * m1) * (Kt_XX_2_sum + Kt_YY_2_sum)
        - 1 / (m * m1) ** 2 * (Kt_XX_sum ** 2 + Kt_YY_sum ** 2)
        + 2 / (m * m) * K_XY_2_sum
        - 2 / m ** 4 * K_XY_sum ** 2
        - 4 / (m * m * m1) * (dot_XX_XY + dot_YY_YX)
        + 4 / (m ** 3 * m1) * (Kt_XX_sum + Kt_YY_sum) * K_XY_sum
    )
    var_est = (4 * (var_at_m - 2) / (var_at_m * (var_at_m - 1)) * zeta1_est
               + 2 / (var_at_m * (var_at_m - 1)) * zeta2_est)

    return mmd2, var_est  # Return MMD2 and its variance