import argparse
import os
import time
import torch
from torchvision.utils import save_image
from utils import visualize_images, epoch_calculate_fretchet, epoch_calculate_activations, epoch_compute_mmd_simple
from models.sb_train import *
from utils.inception_v3 import InceptionV3 as inception_v3
from options.train_options import train_parser
from preprocessing.dataset import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':

    # Model 
    sb_model_compl = SBModel().to(device)
    
    # Parse the arguments
    args = train_parser.parse_args()
    
    # create output images directory
    generated_images = os.makedirs(args.create_dir, exist_ok=True)

    # use args to access the arguments
    total_iters = args.total_iters
    optimize_time = args.optimize_time
    epoch_count = args.epoch_count
    n_epochs = args.n_epochs
    n_epochs_decay = args.n_epochs_decay
    print_freq = args.print_freq
    gpu_ids = args.gpu_ids
    BATCH_SIZE = args.batch_size
    dataroot = args.dataroot
    path_trainA = args.path_trainA
    path_trainB = args.path_trainB
    path_testA = args.path_testA
    path_testB = args.path_testB 
    
    # Initialize datasets for training and testing using the predefined ImageDataset class.
    train_datasetA = ImageDataset(img_dir=path_trainA)  # Dataset of horse images for training.
    train_datasetB = ImageDataset(img_dir=path_trainB)  # Dataset of zebra images for training.
    test_datasetA = ImageDataset(img_dir=path_testA)    # Dataset of horse images for testing.
    test_datasetB = ImageDataset(img_dir=path_testB)    # Dataset of zebra images for testing.
    
    # Create DataLoaders for the datasets to manage loading and batching of data during training and testing.
    train_dataloaderA = DataLoader(train_datasetA, batch_size=BATCH_SIZE, shuffle=True)  # DataLoader for horse images in training.
    train_dataloaderB = DataLoader(train_datasetB, batch_size=BATCH_SIZE, shuffle=True)  # DataLoader for zebra images in training.
    test_dataloaderA = DataLoader(test_datasetA, batch_size=BATCH_SIZE, shuffle=True)     # DataLoader for horse images in testing.
    test_dataloaderB = DataLoader(test_datasetB, batch_size=BATCH_SIZE, shuffle=True)     # DataLoader for zebra images in testing.
    
    
    # Lists for Losses, FID and KID metrics 
    losses_list = []
    fid_list = []
    kid_list = []
    
    sb_model_compl.train()
    
    # Training 
    times = []
    for epoch in range(epoch_count, n_epochs + n_epochs_decay +1):    
        epoch_start_time = time.time()  
        iter_data_time = time.time()    
        epoch_iter = 0     # the number of training iterations in current epoch, reset to 0 every epoch
         
        
        for i, ((dataA, dataB), (dataA2, dataB2)) in enumerate(zip(zip(train_dataloaderA, train_dataloaderB), zip(train_dataloaderA, train_dataloaderB))):  
            dataA = dataA.to(device)
            dataB = dataB.to(device)
            dataA2 = dataA2.to(device)
            dataB2 = dataB2.to(device)
            
            iter_start_time = time.time()  
            if total_iters % print_freq == 0:
                t_data = iter_start_time - iter_data_time

            batch_size = 1
            total_iters += batch_size
            epoch_iter += batch_size
            if len(gpu_ids) > 0:
                torch.cuda.synchronize()
            optimize_start_time = time.time()
            if epoch == epoch_count and i == 0:
                sb_model_compl.data_dependent_initialize(dataA,dataB, dataA2, dataB2)
            sb_model_compl.set_input(dataA,dataB, dataA2, dataB2)  # unpack data from dataset and apply preprocessing
            sb_model_compl.optimize_parameters()   # calculate loss functions, get gradients, update network weights
            
            torch.cuda.synchronize()
            optimize_time = (time.time() - optimize_start_time) / batch_size * 0.005 + 0.995 * optimize_time
            
            if total_iters % print_freq == 0:    # print training losses and save logging information 
                losses = sb_model_compl.get_current_losses()
                print(losses)
                
                

            iter_data_time = time.time()

        
        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, n_epochs + n_epochs_decay, time.time() - epoch_start_time))
         
        # Visualize and save the generated fake images at the end of each epoch
        sb_model_compl.forward()  
        fake_B_images = sb_model_compl.fake_B
        fake_B2_images = sb_model_compl.fake_B2
        real_A_images = sb_model_compl.real_A
        real_B_images = sb_model_compl.real_B
        visualize_images(fake_B_images, title=f"Generated Zebras at Epoch {epoch}")
        visualize_images(fake_B2_images, title=f"Generated Zebras 2 at Epoch {epoch}")
        
        # Save the generated images
        save_image(fake_B_images, os.path.join(generated_images, f"generated_zebras_epoch{epoch}.png"))
        save_image(fake_B2_images, os.path.join(generated_images, f"generated_zebras_2_epoch{epoch}.png"))
        
        # Compute FID 
        fretchet_dist= epoch_calculate_fretchet(dataB,sb_model_compl.fake_B.to(device),inception_v3)
        print(f'Epoch {epoch}: FID:', fretchet_dist)
        
        # Compute activations for KID per epoch 
        activations_real = epoch_calculate_activations(dataB, inception_v3, cuda=True)
        activations_fake = epoch_calculate_activations(sb_model_compl.fake_B.to(device), inception_v3, cuda=True)

        # Calculate KID
        kid_value = epoch_compute_mmd_simple(activations_real, activations_fake)
        print(f'Epoch {epoch}: KID: {kid_value}')
        
        losses_list.append(losses)
        fid_list.append(fretchet_dist)
        kid_list.append(kid_value)