""" Defining Loss Criterions """

import torch.nn as nn 
import torch 

def criterionNCE(nce_layers):
    """ Used for contrastive learning tasks, calculating loss for feature differences across network layers """
    criterionNCE = []
    for nce_layer in nce_layers:
        criterionNCE.append(nn.CrossEntropyLoss(reduction='none').to(device))
    return criterionNCE

def criterionGAN():
    """ Applied in adversarial training, measuring the difference between real and generated samples """
    return nn.MSELoss().to(device)


class GANLoss(nn.Module):
    """Define Least Squares GAN loss."""

    def __init__(self, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class, whose parameters are float representing labels for real and fake images """
        
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.loss = nn.MSELoss()

    def get_target_tensor(self, prediction, target_is_real):
        """ Creates label tensors filled with the ground truth label, and with the size of the input """
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """ Calculate loss given Discriminator's prediction and ground truth labels """
        target_tensor = self.get_target_tensor(prediction, target_is_real)
        return self.loss(prediction, target_tensor)