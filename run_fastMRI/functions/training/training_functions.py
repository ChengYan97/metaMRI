

import torch
from torch.nn import L1Loss, MSELoss

# Implementation of SSIMLoss
from functions.training.losses import SSIMLoss

# Apply a center crop on the larger image to the size of the smaller.
#from functions.data.transforms import center_crop_to_smallest

# In order to get access to attributes stored in save_checkpoint
from functions.train_utils import save_checkpoint

class Compute_batch_train_loss:
    def __init__(self) -> None:
        self.loss_fct_lookup = {
            'SSIM' : SSIMLoss(),
            'L1' : L1Loss(reduction='sum'),
            'L2' : MSELoss(reduction='sum'),
        }
    
    def get_batch_train_loss(self, hp_exp, output, target, max_value, train_meters):
        train_loss = 0
        for loss in hp_exp['loss_functions']:
            if loss == 'SSIM':
                loss = self.loss_fct_lookup['SSIM'](output, target, data_range=max_value)

                train_meters["train_SSIM"].update(loss.item())

                train_loss += loss
            elif loss == 'L1':
                loss = self.loss_fct_lookup['L1'](output, target) / torch.sum(torch.abs(target))
                train_meters["train_L1"].update(loss.item())
                train_loss += loss
            elif loss == 'L2':
                # L2 loss in the image domain, i.e. directly between network output and target image
                loss = self.loss_fct_lookup['L2'](output, target) / torch.sum(torch.abs(target)**2)
                train_meters["train_L2"].update(loss.item())
                train_loss += loss
            elif loss == 'L2_kspace': 
                # L2 loss in the frequency domain. Actually this function works the same as 'L2'
                loss = self.loss_fct_lookup['L2'](output, target) / torch.sum(torch.abs(target)**2)
                train_meters["train_L2_kspace"].update(loss.item())
                train_loss += loss
            elif loss == 'L1_kspace': 
                # L1 loss in the frequency domain. Actually this function works the same as 'L1'
                loss = self.loss_fct_lookup['L1'](output, target) / torch.sum(torch.abs(target))
                train_meters["train_L1_kspace"].update(loss.item())
                train_loss += loss
            #else:
            #    raise ValueError("Chosen loss function is not implemented.")

            if len(hp_exp['loss_functions']) > 1:
                train_meters['cumulated_loss'].update(train_loss.item())

            return train_loss

#def compute_batch_train_loss(hp_exp, output, target, mean, std, max_value, train_meters):
#    '''
#    Compute train loss for a given batch.
#    I don't think that at the moment this code will work for SSIM loss and batch size larger than 1.
#    '''
#    train_loss = 0
#    for loss in hp_exp['loss_functions']:
#        if loss == 'SSIM':
#            loss_fct = SSIMLoss()
#            mean = mean.unsqueeze(1).unsqueeze(2) 
#            std = std.unsqueeze(1).unsqueeze(2) 
#           #target, output = center_crop_to_smallest(target, output)
#            # When working with the SSIM loss, tensors need to be unnormed as it is done automatically by the NormUnet in the VarNet.
#            output = output * std + mean  
#           target = target * std + mean
#            
#           loss = loss_fct(output.unsqueeze(1), target.unsqueeze(1), data_range=max_value)
#
#            train_meters["train_SSIM"].update(loss.item())#
#
#            train_loss += loss
#        elif loss == 'L1':
#            loss_fct = L1Loss()
#            loss = loss_fct(output, target)
#            train_meters["train_L1"].update(loss.item())
#            train_loss += loss
#        elif loss == 'L2':
#            loss_fct = MSELoss()
#            loss = loss_fct(output, target)
#            train_meters["train_L1"].update(loss.item())
#            train_loss += loss
#        else:
#            raise ValueError("Chosen loss function is not implemented.")#
#
#        if len(hp_exp['loss_functions']) > 1:
#            train_meters['cumulated_loss'].update(train_loss.item())
#
#        return train_loss


#def configure_loss_functions(hp_exp):
#    '''
#    Takes a list of training loss functions and a list of validation loss functions form the configuration in hp_exp.
#    Returns 
#        - a dictionary of training loss functions with key: name_of_loss_functoin and value: function_object
#        - a dictionary of validation loss functions with key: name_of_loss_functoin and value: function_object
#    '''
#    trainloss_functions_dict = {}
#    valloss_funcions_dict = {}##
#
#    loss_functions_lookup = {
#        'SSIM' : SSIMLoss(),
#        'L1' : L1Loss(),
#        'L2' : MSELoss(),
#        'PSNR' : PSNRLoss()
#    }
#
#    for loss in hp_exp['loss_functions']:
#        trainloss_functions_dict['train_'+loss] = loss_functions_lookup[loss]#
#
#    for loss in hp_exp['val_loss_function']:
#        valloss_funcions_dict['val_'+loss] = loss_functions_lookup[loss]#
#
#    return trainloss_functions_dict, valloss_funcions_dict


def configure_optimizers(hp_exp, parameters, optimizer=None):

    if not optimizer:
        if hp_exp['optimizer'] == 'Adam':
            optimizer = torch.optim.Adam(
                params=parameters, 
                lr=hp_exp['lr'], 
                betas=(0.9, 0.999), 
                eps=1e-08, 
                weight_decay=0.0, 
                amsgrad=False
            )
        elif hp_exp['optimizer'] == 'RMSprop':
            optimizer = torch.optim.RMSprop(
                parameters,
                lr=hp_exp['lr'],
                weight_decay=0.0,
            )

    
    if hp_exp['lr_scheduler'] == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer, 
            mode=save_checkpoint.mode, 
            factor=hp_exp['lr_decay_factor'], 
            patience=hp_exp['lr_patience'], 
            threshold=hp_exp['lr_threshold'], 
            threshold_mode='abs', 
            cooldown=0, 
            min_lr=hp_exp['lr_min'], 
            eps=1e-08, 
            verbose=True
            )
    elif hp_exp['lr_scheduler'] == 'MultiStepLR':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer=optimizer, 
            milestones=hp_exp['lr_milestones'], 
            gamma=hp_exp['lr_decay_factor'], 
            last_epoch=- 1, 
            verbose=False
            )

    return optimizer, scheduler