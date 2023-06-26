#%%
import random
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
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


experiment_name = 'E6.3_standard(NMSE)_T8x200_100epoch'

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
TRAINING_EPOCH = 100
num_sample_train = 200
num_sample_val = 100
BATCH_SIZE = 5

# data path
path_train1 = '/cheng/metaMRI/metaMRI/data_dict/E6.3/P/knee_train_PD_Aera_2-9.yaml'
path_train2 = '/cheng/metaMRI/metaMRI/data_dict/E6.3/P/knee_train_PD_Aera_15-22.yaml'
path_train3 = '/cheng/metaMRI/metaMRI/data_dict/E6.3/P/knee_train_PD_Biograph_15-22.yaml'
path_train4 = '/cheng/metaMRI/metaMRI/data_dict/E6.3/P/knee_train_PD_Skyra_15-22.yaml'
path_train5 = '/cheng/metaMRI/metaMRI/data_dict/E6.3/P/knee_train_PDFS_Aera_2-9.yaml'
path_train6 = '/cheng/metaMRI/metaMRI/data_dict/E6.3/P/knee_train_PDFS_Aera_15-22.yaml'
path_train7 = '/cheng/metaMRI/metaMRI/data_dict/E6.3/P/knee_train_PDFS_Biograph_15-22.yaml'
path_train8 = '/cheng/metaMRI/metaMRI/data_dict/E6.3/P/knee_train_PDFS_Skyra_15-22.yaml'

path_val1 = '/cheng/metaMRI/metaMRI/data_dict/E6.3/P/knee_val_PD_Aera_2-9.yaml'
path_val2 = '/cheng/metaMRI/metaMRI/data_dict/E6.3/P/knee_val_PD_Aera_15-22.yaml'
path_val3 = '/cheng/metaMRI/metaMRI/data_dict/E6.3/P/knee_val_PD_Biograph_15-22.yaml'
path_val4 = '/cheng/metaMRI/metaMRI/data_dict/E6.3/P/knee_val_PD_Skyra_15-22.yaml'
path_val5 = '/cheng/metaMRI/metaMRI/data_dict/E6.3/P/knee_val_PDFS_Aera_2-9.yaml'
path_val6 = '/cheng/metaMRI/metaMRI/data_dict/E6.3/P/knee_val_PDFS_Aera_15-22.yaml'
path_val7 = '/cheng/metaMRI/metaMRI/data_dict/E6.3/P/knee_val_PDFS_Biograph_15-22.yaml'
path_val8 = '/cheng/metaMRI/metaMRI/data_dict/E6.3/P/knee_val_PDFS_Skyra_15-22.yaml'


# mask function and data transform
mask_function = create_mask_for_mask_type(mask_type_str = 'random', self_sup = False, 
                    center_fraction = 0.08, acceleration = 4.0, acceleration_total = 3.0)

data_transform_train = UnetDataTransform_norm('multicoil', mask_func = mask_function, use_seed=False, mode='train')
data_transform = UnetDataTransform_norm('multicoil', mask_func = mask_function, use_seed=True, mode='adapt')

# training dataset and data loader
path_train_list = [path_train1, path_train2, path_train3, path_train4, path_train5, path_train6, path_train7, path_train8]
trainset_list = []
for path in path_train_list:
    trainset = SliceDataset(dataset = path, path_to_dataset='', path_to_sensmaps=None, provide_senmaps=False, 
                challenge="multicoil", transform=data_transform_train, use_dataset_cache=True, num_samples= num_sample_train)
    trainset_list.append(trainset)

train_set = torch.utils.data.ConcatDataset(trainset_list)
train_dataloader = torch.utils.data.DataLoader(dataset = train_set, batch_size = BATCH_SIZE,
                shuffle = True, generator = torch.Generator().manual_seed(1), pin_memory = True)
print("Training date number: ", len(train_dataloader.dataset))

# validation dataset and data loader
path_val_list = [path_val1, path_val2, path_val3, path_val4, path_val5, path_val6, path_val7, path_val8]
val_dataset_list = []
for path_val in path_val_list:
    valset = SliceDataset(dataset = path_val, path_to_dataset='', path_to_sensmaps=None, provide_senmaps=False, 
                challenge="multicoil", transform=data_transform, use_dataset_cache=True, num_samples= num_sample_val)
    val_dataset_list.append(valset)

val_set = torch.utils.data.ConcatDataset(val_dataset_list)
val_dataloader = torch.utils.data.DataLoader(dataset = val_set, batch_size = 1, 
                shuffle = False, generator = torch.Generator().manual_seed(1), pin_memory = False)
print("Validation date number: ", len(val_dataloader.dataset))



#%%

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

        loss = lossfn(train_outputs, train_targets) / torch.sum(torch.abs(train_targets)**2)
        
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

        loss = lossfn(val_outputs, val_targets) / torch.sum(torch.abs(val_targets)**2)

        total_val_loss += loss.item()
        
    validation_loss = total_val_loss / len(dataloader)
    return validation_loss


model = Unet(in_chans=1, out_chans=1, chans=32, num_pool_layers=4, drop_prob=0.0)
model = model.to(device)


##########################
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
lossfn = torch.nn.MSELoss(reduction='sum')

best_loss = 10.000
for iteration in range(TRAINING_EPOCH):
    print('Iteration:', iteration+1)
    # training
    training_loss = train(model, train_dataloader, optimizer)
    print('Training NMSE', training_loss) 
    writer.add_scalar("Training NMSE", training_loss, iteration+1)
    # val
    validation_loss = evaluate(model, val_dataloader)
    print('Validation NMSE', validation_loss) 
    writer.add_scalar("Validation NMSE", validation_loss, iteration+1)
    if best_loss > validation_loss:
        best_loss = validation_loss
        save_path = '/cheng/metaMRI/metaMRI/save/'+ experiment_name + '/' + experiment_name + '_E' + str(iteration+1) + '_best.pth'
        torch.save((model.state_dict()), save_path)
        print('Model saved to', save_path)
    else:
        pass
    #scheduler.step()

### save model
# save_path = '/cheng/metaMRI/metaMRI/save/'+ experiment_name + '/' + experiment_name + '.pth'
# torch.save((model.state_dict()), save_path)
# print('Model saved to', save_path)
