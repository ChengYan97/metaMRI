import random
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import torch
import learn2learn as l2l
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
# The corase reconstruction is the rss of the zerofilled multi-coil kspaces
# after inverse FT.
from functions.data.transforms import UnetDataTransform, UnetDataTransform_norm, normalize
# Import a torch.utils.data.Dataset class that takes a list of data examples, a path to those examples
# a data transform and outputs a torch dataset.
from functions.data.mri_dataset import SliceDataset
# Unet architecture as nn.Module
from functions.models.unet import Unet
# Function that returns a MaskFunc object either for generatig random or equispaced masks
from functions.data.subsample import create_mask_for_mask_type
# Implementation of SSIMLoss
from functions.training.losses import SSIMLoss

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


experiment_name = 'E6.2_standard_NMAE_Q3'


# tensorboard dir
experiment_path = '/cheng/metaMRI/metaMRI/save/' + experiment_name + '/'
writer = SummaryWriter(experiment_path)

# seed
SEED = 1
random.seed(SEED)
np.random.seed(SEED)
torch.cuda.manual_seed(SEED)
torch.manual_seed(SEED)

# hyperparameter
TRAINING_EPOCH = 70
num_sample_train = 300
num_sample_val = 100

path_train1 = '/cheng/metaMRI/metaMRI/data_dict/E6.2/brain_train_AXT2_Aera_5-8.yaml'
# '/cheng/metaMRI/metaMRI/data_dict/E6.2/brain_train_AXT1POST_TrioTim_5-8.yaml'
# '/cheng/metaMRI/metaMRI/data_dict/E6.2/brain_train_AXFLAIR_Skyra_5-8.yaml'
# '/cheng/metaMRI/metaMRI/data_dict/E6.2/brain_train_AXT2_Aera_5-8.yaml'

path_val1 = '/cheng/metaMRI/metaMRI/data_dict/E6.2/brain_val_AXT2_Aera_5-8.yaml'
# '/cheng/metaMRI/metaMRI/data_dict/E6.2/brain_val_AXT1POST_TrioTim_5-8.yaml'
# '/cheng/metaMRI/metaMRI/data_dict/E6.2/brain_val_AXFLAIR_Skyra_5-8.yaml'
# '/cheng/metaMRI/metaMRI/data_dict/E6.2/brain_val_AXT2_Aera_5-8.yaml'

# mask function and data transform
mask_function = create_mask_for_mask_type(mask_type_str = 'random', self_sup = False, 
                    center_fraction = 0.08, acceleration = 4.0, acceleration_total = 3.0)

data_transform_train = UnetDataTransform_norm('multicoil', mask_func = mask_function, use_seed=False, mode='train')
data_transform = UnetDataTransform_norm('multicoil', mask_func = mask_function, use_seed=True, mode='adapt')

# dataset: num_sample_subset x 3
trainset1 = SliceDataset(dataset = path_train1, path_to_dataset='', path_to_sensmaps=None, provide_senmaps=False, 
                        challenge="multicoil", transform=data_transform_train, use_dataset_cache=True, num_samples= num_sample_train)
#trainset2 = SliceDataset(dataset = path_train2, path_to_dataset='', path_to_sensmaps=None, provide_senmaps=False, 
#                        challenge="multicoil", transform=data_transform_train, use_dataset_cache=True, num_samples= num_sample_train)
trainset = torch.utils.data.ConcatDataset([trainset1])

print("Training date number: ", len(trainset))
valset1 = SliceDataset(dataset = path_val1, path_to_dataset='', path_to_sensmaps=None, provide_senmaps=False, 
                      challenge="multicoil", transform=data_transform, use_dataset_cache=True, num_samples= num_sample_val)
#valset2 = SliceDataset(dataset = path_val2, path_to_dataset='', path_to_sensmaps=None, provide_senmaps=False, 
#                      challenge="multicoil", transform=data_transform, use_dataset_cache=True, num_samples= num_sample_val)
valset = torch.utils.data.ConcatDataset([valset1])
print("Validation date number: ", len(valset))

# dataloader: batch size 1 
train_dataloader = torch.utils.data.DataLoader(dataset = trainset, batch_size = 8, num_workers = 8, 
                    shuffle = True, generator = torch.Generator().manual_seed(1), pin_memory = True)
val_dataloader = torch.utils.data.DataLoader(dataset = valset, batch_size = 1, num_workers = 8, 
                    shuffle = False, generator = torch.Generator().manual_seed(1))


def train(model, dataloader, optimizer): 
    model.train()
    train_loss = 0.0

    for iter, batch in tqdm(enumerate(dataloader)):
        input_image, target_image, mean, std, fname, slice_num = batch
        train_inputs = input_image.to(device)
        train_targets = target_image.to(device)
        std = std.to(device)
        mean = mean.to(device)

        train_outputs = model(train_inputs)
        train_outputs = train_outputs * std + mean

        loss = l1_loss(train_outputs, train_targets) / torch.sum(torch.abs(train_targets))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    avg_train_loss = train_loss / len(dataloader)
    return avg_train_loss


def evaluate(model, dataloader):
    model.eval()
    total_val_loss = 0.0
    
    for iter, batch in tqdm(enumerate(dataloader)): 
        input_image, target_image, mean, std, fname, slice_num = batch
        val_inputs = input_image.to(device)
        val_targets = target_image.to(device)
        std = std.to(device)
        mean = mean.to(device)

        val_outputs = model(val_inputs)
        val_outputs = val_outputs * std + mean

        # SSIM = 1 - loss
        loss = l1_loss(val_outputs, val_targets) / torch.sum(torch.abs(val_targets))

        total_val_loss += loss.item()
        
    validation_loss = total_val_loss / len(dataloader)
    return validation_loss


model = Unet(in_chans=1, out_chans=1, chans=32, num_pool_layers=4, drop_prob=0.0)
model = model.to(device)


##########################
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
l1_loss = torch.nn.L1Loss(reduction='sum')

best_loss = 10.000
for iteration in range(TRAINING_EPOCH):
    print('Iteration:', iteration+1)
    # training
    training_loss = train(model, train_dataloader, optimizer)
    print('Training NMAE', training_loss) 
    writer.add_scalar("Training NMAE", training_loss, iteration+1)
    # val
    validation_loss = evaluate(model, val_dataloader)
    print('Validation NMAE', validation_loss) 
    writer.add_scalar("Validation NMAE", validation_loss, iteration+1)
    if best_loss > validation_loss:
        best_loss = validation_loss
        save_path = '/cheng/metaMRI/metaMRI/save/'+ experiment_name + '/' + experiment_name + '_E' + str(iteration+1) + '_best.pth'
        torch.save((model.state_dict()), save_path)
        print('Model saved to', save_path)
    else:
        pass
    #scheduler.step()

### save model
save_path = '/cheng/metaMRI/metaMRI/save/'+ experiment_name + '/' + experiment_name + '.pth'
torch.save((model.state_dict()), save_path)
print('Model saved to', save_path)
