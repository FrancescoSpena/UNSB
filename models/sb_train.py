from models.networks import *
from models.helper_functions import *
from utils.loss_criterions import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

""" Sb Model defined for training. Generator, Discriminator and PatchSample (namely gen, disc, netE and netF are defined in networks.py)"""

class SBModel(nn.Module):
    
    def __init__(self):
        """ Initializes the SBModel class, setting up parameters, loss names, model names, visual names, optimizers, and other necessary configurations """
        # Note that parameters have been taken directly from the paper, except for the number of epochs, due to our hardware computation limits 
        super(SBModel,self).__init__()
        self.loss_names = ['G_GAN', 'D_real', 'D_fake', 'G', 'NCE','SB']
        self.model_names = ['G','F','D','E']
        self.visual_names = ['real_A','real_A_noisy', 'fake_B', 'real_B']
        self.optimizers = []
        self.tau = 0.1
        self.device = device   
        self.lambda_NCE = 1.0 
        self.nce_idt = True
        self.nce_layers = [0,4,8,12,16]  
        self.num_patches = 256
        self.netG = gen
        self.netD = disc
        self.netE = netE
        self.netF = netF 
        self.ngf = 64
        self.criterionNCE = criterionNCE(self.nce_layers)
        self.criterionGAN = GANLoss().to(device)
        self.lr = 0.00001
        self.beta1 = 0.5
        self.beta2 = 0.999 
        
        # Defining Optimizers
        self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))
        self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))
        self.optimizer_E = torch.optim.Adam(self.netE.parameters(),lr=self.lr, betas=(self.beta1, self.beta2))
        
    def data_dependent_initialize(self, dataA,dataB, dataA2, dataB2): 
        """ Prepares the model for training, computing fake images using the generator and initial losses for the generator 'G' 
        and the discriminators 'D' and 'E'. It is conditioned on whether the loss function involving the NCE term is active. 
        If so, it initializes an optimizer for netF """
        bs = 1
        self.set_input(dataA,dataB, dataA2, dataB2)
        self.real_A = self.real_A[:bs]
        self.real_B = self.real_B[:bs]
        self.real_A2 = self.real_A2[:bs]
        self.real_B2 = self.real_B2[:bs]
        self.forward()  
        self.compute_G_loss().backward()
        self.compute_D_loss().backward()
        self.compute_E_loss().backward()  
        if self.lambda_NCE > 0.0:
            self.optimizer_F = torch.optim.Adam(self.netF.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))
            self.optimizers.append(self.optimizer_F)
        
    
    def set_input(self, dataA, dataB, dataA2, dataB2):
        """ Responsible for unpacking input data from the dataloader and performing any necessary preprocessing steps """
        self.real_A = dataA.to(device)
        self.real_B = dataB.to(device)
        self.real_A2 = dataA2.to(device)
        self.real_B2 = dataB2.to(device)
        
    def set_requires_grad(self, nets, requires_grad=True):
        """ Toggles the requirement for gradient computation for the parameters in the provided networks. 
        It is s helpful for controlling which parts of the model are trainable during different training phases """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
    
    def get_current_losses(self):
        """ Retrieves the current training losses/errors from the model and returns them as a dictionary """
        errors_ret = {}
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, 'loss_' + name))  # float(...) works for both scalar tensor and float number
        return errors_ret
        
    def optimize_parameters(self):
        """ It ensures that the gradients are properly calculated and used to update the network weights during training """
        
        # Forward pass
        self.forward()
        
        # Set models to training mode
        self.netG.train()
        self.netE.train()
        self.netD.train()
        self.netF.train()
        
        # Update Discriminator D 
        self.set_requires_grad(self.netD, True) # Enables gradient calculation for D
        self.optimizer_D.zero_grad()  # Zeros the gradients of D_optimizer
        self.loss_D = self.compute_D_loss()  # Computes D loss 
        self.loss_D.backward()  # Backpropagates the gradient 
        torch.nn.utils.clip_grad_norm_(self.netD.parameters(), max_norm=1)   # Clip gradients to avoid exploiding gradients 
        self.optimizer_D.step()  # Updates the parameters of D 
        
        # Update Discriminator E
        self.set_requires_grad(self.netE, True)  # Enables gradient calculation for E
        self.optimizer_E.zero_grad()  # Zeros the gradients of E_optimizer
        self.loss_E = self.compute_E_loss()  # Computes E loss
        self.loss_E.backward()   # Backpropagates the gradient 
        torch.nn.utils.clip_grad_norm_(self.netE.parameters(), max_norm=1)  # Clip gradients to avoid exploiding gradients 
        self.optimizer_E.step()  # Updates the parameters of E
    
        # Update Generator G
        self.set_requires_grad(self.netD, False)  # Disables gradient calculation for discriminator D since it is not being updated in this step  
        self.set_requires_grad(self.netE, False)  # Disables gradient calculation for discriminator E since it is not being updated in this step  
        
        self.optimizer_G.zero_grad()  # Zeros the gradient of G_optimizer
        self.optimizer_F.zero_grad()  # Zeros the gradient of F_optimizer 
        
        self.loss_G = self.compute_G_loss()   # Compute G loss 
        self.loss_G.backward()  # Backpropagates the gradient
        
        torch.nn.utils.clip_grad_norm_(self.netG.parameters(), max_norm=1)  # Clip gradients to avoid exploiding gradients 
        self.optimizer_G.step()  # Updates the parameters of G
    
        torch.nn.utils.clip_grad_norm_(self.netF.parameters(), max_norm=1)  #  Clip gradients to avoid exploiding gradients  
        self.optimizer_F.step()  # Updates the parameters of F

    def forward(self):
        """ The diffusion process gradually adds noise to the images, which is crucial for generating diverse and realistic intermediate 
        states that the generator network refines.
        By interpolating between previous states and adding controlled noise, we ensure smooth transitions and avoid abrupt changes in the image states.
        Using the generator network at each timestep allows us to iteratively improve the quality of the noisy images, aligning them closer to the desired 
        output distribution """
        
        tau = 0.01  # Entropy parameter 
        T = 5  # Number of time steps 
        incs = np.array([0] + [1/(i+1) for i in range(T-1)])  # Array of incremental values used to define time steps 
        times = np.cumsum(incs)
        times = times / times[-1]
        times = 0.5 * times[-1] + 0.5 * times
        times = torch.tensor(times).float().cuda()   # Array of normalized time steps, scaled and shifted 
        self.times = times
        bs =  self.real_A.size(0)
        time_idx = (torch.randint(T, size=[1]).cuda() * torch.ones(size=[1]).cuda()).long()  # Randomly selected time step index 
        self.time_idx = time_idx  
        self.timestep     = times[time_idx]  # Actual time value corresponsing to 'time_idx'
        
        with torch.no_grad():
            self.netG.eval()
            for t in range(0, self.time_idx.int().item() + 1):  # Iteration over each time step up to the current index 
                if t > 0:
                    # Interpolation factors based on the current and previous time steps for temporal interpolation -> Paper Fig. 3 
                    delta = times[t] - times[t-1]   
                    denom = times[-1] - times[t-1]  
                    inter = (delta / denom)         
                    scale = (delta * (1 - delta / denom))  
                    
                
                """ Handling Input 1 """
                Xt       = self.real_A if (t == 0) else (1-inter)* Xt + inter * Xt_1.detach() + (scale * tau).sqrt() * torch.randn_like(Xt).to(self.real_A.to(device))
                # Xt is updated using its previous state, the output from the previous timestep Xt_1, and added Gaussian noise 
                time_idx = (t * torch.ones(size=[self.real_A.shape[0]]).to(self.real_A.to(device))).long()
                time     = times[time_idx]
                z        = torch.randn(size=[self.real_A.shape[0],4*self.ngf]).to(self.real_A.to(device))
                Xt_1     = self.netG(Xt, time_idx, z) # Xt_1 is the output of the generator given the noisy input Xt, current time_idx, and latent vector z. 
        
                """ Handling input 2 """
                # We consider another input to help stabilize the training process. This ensures that the model learns consistent features across different instances. It can be considered as a sort of data augmentation 
                Xt2 = self.real_A2 if (t == 0) else (1-inter)*Xt2 + inter*Xt_12.detach() + (scale*tau).sqrt() * torch.randn_like(Xt2).to(self.real_A2.to(device))
                # Xt2 is updated using its previous state, the output from the previous timestep Xt_12, and added Gaussian noise 
                time_idx = (t * torch.ones(size=[self.real_A.shape[0]]).to(self.real_A.to(device))).long()
                time     = times[time_idx]
                z        = torch.randn(size=[self.real_A.shape[0], 4 * self.ngf]).to(self.real_A.to(device))
                Xt_12    = self.netG(Xt2, time_idx, z)  # Xt_12 is the output of the generator given the noisy input Xt2, current time_idx, and latent vector z.
                
                if self.nce_idt:
                    XtB = self.real_B if (t == 0) else (1-inter) * XtB + inter * Xt_1B.detach() + (scale * tau).sqrt() * torch.randn_like(XtB).to(self.real_A.to(device))
                    # XtB is updated using its previous state, the output from the previous timestep Xt_1B, and added Gaussian noise 
                    time_idx = (t * torch.ones(size=[self.real_A.shape[0]]).to(self.real_A.to(device))).long()
                    time     = times[time_idx]
                    z        = torch.randn(size=[self.real_A.shape[0],4*self.ngf]).to(self.real_A.to(device))
                    Xt_1B = self.netG(XtB, time_idx, z)  # Xt_1B is the output of the generator given the noisy input XtB, current time_idx, and latent vector z.
                    
            if self.nce_idt:
                self.XtB = XtB.detach()
                
            self.real_A_noisy = Xt.detach()
            self.real_A_noisy2 = Xt2.detach()          
        
        z_in    = torch.randn(size=[2*bs,4*self.ngf]).to(self.real_A.to(device))  # Random noise for generator inputs 
        z_in2    = torch.randn(size=[bs,4*self.ngf]).to(self.real_A.to(device))   # # Random noise for generator inputs 
        
        self.real = torch.cat((self.real_A, self.real_B), dim=0) if self.nce_idt else self.real_A   # Concatenate real horse and real zebra image if identity loss is enabled 
        self.realt = torch.cat((self.real_A_noisy, self.XtB), dim=0) if self.nce_idt else self.real_A_noisy  # Concatenate noisy horse and noisy zebra if identity loss is enabled 
        self.fake = self.netG(self.realt,self.time_idx,z_in)   # Apply the generator to the concatenated first step of  noisy images 
        self.fake_B2 =  self.netG(self.real_A_noisy2,self.time_idx,z_in2)  # Apply the generator to the second set of noisy image 
        self.fake_B = self.fake[:self.real_A.size(0)]  # Extract "generated zebra" (horse with zebra's features) from self.fake   

    def compute_D_loss(self):
        """ It calculates the adversarial loss for the discriminator. It computes separate losses for real and fake images and then 
        combines them to obtain the total discriminator loss. The latter is scaled by 0.5 to ensure equal contribution from both 
        fake and real losses. This loss guides the training of the discriminator to better distinguish between real and generated images. """
        
        bs = self.real_A.size(0)
        fake = self.fake_B.detach()   # Obtained Fake Images 
        std = torch.rand(size=[1]).item()
        pred_fake = self.netD(fake,self.time_idx)    # Discriminator D's predictions for fake images 
        self.loss_D_fake = self.criterionGAN(pred_fake, False).mean() #Computes Adversarial Loss for fake images, setting target label to 'False' to denote that the images are fake
        self.pred_real = self.netD(self.real_B,self.time_idx)  # Discriminator D's predictions for real images 
        loss_D_real = self.criterionGAN(self.pred_real, True)  # Computes Adversarial Loss for real images, setting target label to 'True' to denote that the images are real
        self.loss_D_real = loss_D_real.mean()
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5  # Total Discriminator D loss 
        return self.loss_D
    
    def compute_E_loss(self):
        """ Computation of Discriminator E loss, whose primary goal is to guide the training of netE towards learning meaningful 
        representations of transition distributions between noisy and generated image pairs. By minimizing the loss and incorporating 
        regularization techniques, the network aims to align these distributions effectively, facilitating the generation of realistic 
        and coherent images by the generator network G """
        
        bs =  self.real_A.size(0)
        XtXt_1 = torch.cat([self.real_A_noisy,self.fake_B.detach()], dim=1)   # Concatenation of noisy input image with the corresponding generated image 
        XtXt_2 = torch.cat([self.real_A_noisy2,self.fake_B2.detach()], dim=1) # Concatenation of noisy input image 2 with the corresponding generated image 2 
        temp = torch.logsumexp(self.netE(XtXt_1, self.time_idx, XtXt_2).reshape(-1), dim=0).mean()  # Entropy term which includes a log-sum-exp term for stability
        # This operation helps to approximate the log of the integral of the transition probabilities, providing a more stable and robust computation 
        self.loss_E = -self.netE(XtXt_1, self.time_idx, XtXt_1).mean() +temp + temp**2  # Total E loss is computed, including terms related to negative LL and regularization 
        
        return self.loss_E
    
    def compute_G_loss(self):
        """ It evaluates the overall loss incurred by, encompassing multiple loss components, such as : 
              ** G_GAN Loss ** : Encourages the generator to produce realistic images 
              ** Schrödinger Bridge Loss ** : Ensures temporal consistency and distribution alignment 
              ** Negative Cross Entropy ** : Promotes feature alignment between real and generated images
            This combination enables the generator to learn effective image generation strategies that produce high-quality images 
            consistent with the target distribution """
        
        bs = 1
        tau = 0.01
        lambda_GAN = 1.0
        lambda_SB = 1.0
        lambda_NCE = 1.0
        
        fake = self.fake_B
        std = torch.rand(size=[1]).item() 
        
        if lambda_GAN > 0:
            pred_fake = self.netD(fake,self.time_idx)  # Discriminator D predictions on generated images 
            self.loss_G_GAN = self.criterionGAN(pred_fake, True).mean() # Compares fake predictions to the target label 'True', indicating these should be real 
        else:
            self.loss_G_GAN = 0

        self.loss_SB = 0
        if lambda_SB > 0:
            XtXt_1 = torch.cat([self.real_A_noisy, self.fake_B], dim=1)
            XtXt_2 = torch.cat([self.real_A_noisy2, self.fake_B2], dim=1)
            bs = 1
            ET_XY    = self.netE(XtXt_1, self.time_idx, XtXt_1).mean() - torch.logsumexp(self.netE(XtXt_1, self.time_idx, XtXt_2).reshape(-1), dim=0)  # Helps in aligning the distributions 
            self.loss_SB = -(self.timestep - self.time_idx[0])/self.timestep*tau*ET_XY
            self.loss_SB += self.tau*torch.mean((self.real_A_noisy-self.fake_B)**2)
        
        if lambda_NCE > 0:
            self.loss_NCE = self.calculate_NCE_loss(self.real_A, fake, lambda_NCE) # NCE loss helps in aligning the feature distributions of the real and generated images 
        else: 
            self.loss_NCE, self.loss_NCE_bd = 0.0, 0.0

        self.loss_G = lambda_GAN * self.loss_G_GAN + lambda_SB * self.loss_SB + lambda_NCE * self.loss_NCE # Total Generator loss 
        return self.loss_G
        
    def calculate_NCE_loss(self, src, tgt, lambda_NCE):
        """ Computation of Noise Contrastive Estimation Loss, measuring similarity between patches extracted from source and target images. 
        After applying a weighting factor, it returns the average loss across all layers """
        
        num_patches = 256
        nce_layers = [0,4,8,12,16]
        num_layers = len(nce_layers)
        z = torch.randn(size=[self.real_A.size(0),4*self.ngf]).to(self.real_A.to(device))
        feat_q = self.netG(tgt, self.time_idx, z)  # Feature Map obtained from the generator for target images 
        feat_k = self.netG(src, self.time_idx,z)   # Feature Map obtained from the generator for source images 
        feat_k_pool, sample_ids = self.netF(feat_k, num_patches, None)  # Through netF, we extract patches from the feature maps for target images 
        feat_q_pool, _ = self.netF(feat_q, num_patches, sample_ids)     # Through netF, we extract patches from the feature maps for source images 

        total_nce_loss = 0.0
        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionNCE, nce_layers):
            loss = crit(f_q, f_k) * lambda_NCE   # Cross entropy is used to measure the similarity between the patches 
            total_nce_loss += loss.mean()
        return total_nce_loss / num_layers  # Total loss is averaged across all sampled patches and layers  
    
