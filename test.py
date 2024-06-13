import os
import torch
from torchvision.utils import save_image
from models.helper_functions import *
from models.networks import *
from preprocessing.dataset import *
from models.sb_test import *
from inception import InceptionV3 as inception_v3
from options.test_options import test_parser
from utils.FID_dataset import *
from utils.FID_epoch import *
from utils.loss_criterions import *
from utils.KID_dataset import *
from utils.KID_epoch import *


if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    args = test_parser.parse_args()
    
    # create output images directory
    results_dir = os.makedirs(args.create_dir, exist_ok=True)
    
    dataroot = args.dataroot 
    path_testA = args.path_testA
    path_testB = args.path_testB 
    num_threads = args.num_threads
    BATCH_SIZE = args.batch_size
    serial_batches = args.serial_batches
    no_flip = args.no_flip
    aspect_ratio = args.aspect_ratio
    
    # Defining Dataset and Dataloader 
    test_datasetA = ImageDataset(img_dir=path_testA)    # Dataset of horse images for testing.
    test_datasetB = ImageDataset(img_dir=path_testB)    # Dataset of zebra images for testing.
    test_dataloaderA = DataLoader(test_datasetA, batch_size=BATCH_SIZE, shuffle=True)     # DataLoader for horse images in testing.
    test_dataloaderB = DataLoader(test_datasetB, batch_size=BATCH_SIZE, shuffle=True)     # DataLoader for zebra images in testing.
    
    sb_model_test = SBModel_test().to(device)
    
    pretrained_dict = torch.load("/models/our_pretrained_sb_model_400_epochs.pth")
    model_dict = sb_model_test.state_dict()
    
    fid_list_test = []
    kid_list_test = []

    # Filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
    # Update the new model's dict with the pretrained dict
    model_dict.update(pretrained_dict)  
    sb_model_test.load_state_dict(model_dict)
    sb_model_test.eval()
    
    for i, (dataA, dataB) in enumerate(zip(test_dataloaderA, test_dataloaderB)):
        dataA = dataA.to(device)
        dataB = dataB.to(device)
        if i == 0:
            sb_model_test.data_dependent_initialize(dataA, dataB)
            sb_model_test.eval()
            
        # Unpack data from data loader
        sb_model_test.set_input(dataA, dataB)  
        # Run inference
        sb_model_test.forward()  
        
        fake_B_images = sb_model_test.Xt_1
        visualize_images(fake_B_images.to(device), title="Generated Zebras")
        
        # Save the generated images if needed
        save_image(fake_B_images, os.path.join(results_dir, f"generated_zebras_{i}.png"))
        
        #Compute FID 
        fretchet_dist = epoch_calculate_fretchet(dataB, fake_B_images.to(device), inception_v3)
        print('FID:', fretchet_dist)
        
        # Compute activations
        activations_real = epoch_calculate_activations(dataB, inception_v3, cuda=True)
        activations_fake = epoch_calculate_activations(fake_B_images.to(device), inception_v3, cuda=True)
        
        # Calculate KID
        kid_value = epoch_compute_mmd_simple(activations_real, activations_fake)
        print('KID:', kid_value)
        
        fid_list_test.append(fretchet_dist)
        kid_list_test.append(kid_value)
        