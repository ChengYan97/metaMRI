import random
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import copy
import torch
import learn2learn as l2l
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
# The corase reconstruction is the rss of the zerofilled multi-coil kspaces
# after inverse FT.
from functions.data.transforms import UnetDataTransform, normalize
# Import a torch.utils.data.Dataset class that takes a list of data examples, a path to those examples
# a data transform and outputs a torch dataset.
from functions.data.mri_dataset import SliceDataset
# Unet architecture as nn.Module
from functions.models.unet import Unet
# Function that returns a MaskFunc object either for generatig random or equispaced masks
from functions.data.subsample import create_mask_for_mask_type

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

experiment_name = 'E5.1_pretrain_T9x100_100epoch'

# tensorboard dir
experiment_path = '/cheng/metaMRI/metaMRI/save/' + experiment_name + '/'
writer = SummaryWriter(experiment_path)

TRAINING_EPOCH = 100

########################### datas  ###########################
num_train_subset = 100
num_val_subset = 300
fewshot = 10

# data path
train_path1 = '/cheng/metaMRI/metaMRI/data_dict_narrow/brain_train_AXT1_Skyra.yaml'
train_path2 = '/cheng/metaMRI/metaMRI/data_dict_narrow/brain_train_AXT2_Avanto.yaml'
train_path3 = '/cheng/metaMRI/metaMRI/data_dict_narrow/brain_train_AXT2_Aera.yaml'
train_path4 = '/cheng/metaMRI/metaMRI/data_dict_narrow/brain_train_AXT1POST_Aera.yaml'
train_path5 = '/cheng/metaMRI/metaMRI/data_dict_narrow/brain_train_AXT1POST_Prisma_fit.yaml'
train_path6 = '/cheng/metaMRI/metaMRI/data_dict_narrow/brain_train_AXFLAIR_Biograph_mMR.yaml'
train_path7 = '/cheng/metaMRI/metaMRI/data_dict_narrow/brain_train_AXFLAIR_Skyra.yaml'
train_path8 = '/cheng/metaMRI/metaMRI/data_dict_narrow/brain_train_AXT1PRE_Prisma_fit.yaml'
train_path9 = '/cheng/metaMRI/metaMRI/data_dict_narrow/brain_train_AXT1PRE_TrioTim.yaml'

in_tuning_path = '/cheng/metaMRI/metaMRI/data_dict_narrow/brain_train_AXT2_Aera.yaml'
out_tuning_path = '/cheng/metaMRI/metaMRI/data_dict_narrow/knee_train_CORPD_FBK_Skyra.yaml'
in_val_path = '/cheng/metaMRI/metaMRI/data_dict_narrow/brain_val_AXT2_Aera.yaml'
out_val_path = '/cheng/metaMRI/metaMRI/data_dict_narrow/knee_val_CORPD_FBK_Skyra.yaml'

# mask function and data transform
mask_function = create_mask_for_mask_type(mask_type_str = 'random', self_sup = False, 
                    center_fraction = 0.08, acceleration = 4.0, acceleration_total = 3.0)

data_transform_train = UnetDataTransform('multicoil', mask_func = mask_function, use_seed=False)
data_transform_test = UnetDataTransform('multicoil', mask_func = mask_function, use_seed=True)

# dataset
trainset_1 =  SliceDataset(dataset = train_path1, path_to_dataset='', path_to_sensmaps=None, provide_senmaps=False, 
                challenge="multicoil", transform=data_transform_train, use_dataset_cache=True, num_samples= num_train_subset)
trainset_2 =  SliceDataset(dataset = train_path2, path_to_dataset='', path_to_sensmaps=None, provide_senmaps=False, 
                challenge="multicoil", transform=data_transform_train, use_dataset_cache=True, num_samples= num_train_subset)
trainset_3 =  SliceDataset(dataset = train_path3, path_to_dataset='', path_to_sensmaps=None, provide_senmaps=False, 
                challenge="multicoil", transform=data_transform_train, use_dataset_cache=True, num_samples= num_train_subset)
trainset_4 =  SliceDataset(dataset = train_path4, path_to_dataset='', path_to_sensmaps=None, provide_senmaps=False, 
                challenge="multicoil", transform=data_transform_train, use_dataset_cache=True, num_samples= num_train_subset)
trainset_5 =  SliceDataset(dataset = train_path5, path_to_dataset='', path_to_sensmaps=None, provide_senmaps=False, 
                challenge="multicoil", transform=data_transform_train, use_dataset_cache=True, num_samples= num_train_subset)
trainset_6 =  SliceDataset(dataset = train_path6, path_to_dataset='', path_to_sensmaps=None, provide_senmaps=False, 
                challenge="multicoil", transform=data_transform_train, use_dataset_cache=True, num_samples= num_train_subset)
trainset_7 =  SliceDataset(dataset = train_path7, path_to_dataset='', path_to_sensmaps=None, provide_senmaps=False, 
                challenge="multicoil", transform=data_transform_train, use_dataset_cache=True, num_samples= num_train_subset)
trainset_8 =  SliceDataset(dataset = train_path8, path_to_dataset='', path_to_sensmaps=None, provide_senmaps=False, 
                challenge="multicoil", transform=data_transform_train, use_dataset_cache=True, num_samples= num_train_subset)
trainset_9 =  SliceDataset(dataset = train_path9, path_to_dataset='', path_to_sensmaps=None, provide_senmaps=False, 
                challenge="multicoil", transform=data_transform_train, use_dataset_cache=True, num_samples= num_train_subset)
trainset = torch.utils.data.ConcatDataset([trainset_1, trainset_2, trainset_3, trainset_4, trainset_5, 
                                           trainset_6, trainset_7, trainset_8, trainset_9])

in_tuning_set = SliceDataset(dataset = in_tuning_path, path_to_dataset='', path_to_sensmaps=None, provide_senmaps=False, 
                challenge="multicoil", transform=data_transform_train, use_dataset_cache=True, num_samples= fewshot)
out_tuning_set = SliceDataset(dataset =  out_tuning_path, path_to_dataset='', path_to_sensmaps=None, provide_senmaps=False, 
                challenge="multicoil", transform=data_transform_train, use_dataset_cache=True, num_samples= fewshot)

in_val_set =  SliceDataset(dataset = in_val_path, path_to_dataset='', path_to_sensmaps=None, provide_senmaps=False, 
                challenge="multicoil", transform=data_transform_train, use_dataset_cache=True, num_samples= num_val_subset)
out_val_set =  SliceDataset(dataset = out_val_path, path_to_dataset='', path_to_sensmaps=None, provide_senmaps=False, 
                challenge="multicoil", transform=data_transform_train, use_dataset_cache=True, num_samples= num_val_subset)


# dataloader
train_dataloader = torch.utils.data.DataLoader(dataset = trainset, batch_size = 1, num_workers = 0, 
                    shuffle = True, generator = torch.Generator().manual_seed(1), pin_memory = True)

in_tuning_dataloader = torch.utils.data.DataLoader(dataset = in_tuning_set, batch_size = 1, num_workers = 0, 
                    shuffle = True, generator = torch.Generator().manual_seed(1), pin_memory = False)
out_tuning_dataloader = torch.utils.data.DataLoader(dataset = out_tuning_set, batch_size = 1, num_workers = 0, 
                    shuffle = True, generator = torch.Generator().manual_seed(1), pin_memory = False)
in_validation_dataloader = torch.utils.data.DataLoader(dataset = in_val_set, batch_size = 1, num_workers = 0, 
                    shuffle = False, generator = torch.Generator().manual_seed(1), pin_memory = False)
out_validation_dataloader = torch.utils.data.DataLoader(dataset = out_val_set, batch_size = 1, num_workers = 0, 
                    shuffle = False, generator = torch.Generator().manual_seed(1), pin_memory = False)


def train(model, dataloader, optimizer): 
    model.train()
    train_loss = 0.0
    lossfn = torch.nn.MSELoss(reduction='sum')

    for iter, batch in enumerate(dataloader):
        train_inputs = batch.image.unsqueeze(1).to(device)
        train_targets = batch.target.unsqueeze(1).to(device)

        train_outputs = model(train_inputs)

        loss = lossfn(train_outputs, train_targets) / torch.sum(torch.abs(train_targets)**2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    avg_train_loss = train_loss / len(dataloader.dataset)
    return avg_train_loss

def tuning(model, dataloader, epoch): 
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
    lossfn = torch.nn.MSELoss(reduction='sum')
    for iteration in range(epoch):
        total_loss = 0.0
        for iter, batch in enumerate(dataloader): 
            inputs = batch.image.unsqueeze(1).to(device)
            targets = batch.target.unsqueeze(1).to(device)
            output = model(inputs)
            loss = lossfn(output, targets) / torch.sum(torch.abs(targets)**2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model

def evaluate(model, dataloader):
    model.eval()
    total_loss = 0.0
    lossfn = torch.nn.MSELoss(reduction='sum')
    for iter, batch in enumerate(dataloader): 
        inputs = batch.image.unsqueeze(1).to(device)
        targets = batch.target.unsqueeze(1).to(device)
        output = model(inputs)
        # NMSE
        loss = lossfn(output, targets).item() / torch.sum(torch.abs(targets)**2)
        total_loss += loss
    evaluate_loss = total_loss / len(dataloader.dataset)
    return evaluate_loss

model = Unet(in_chans=1, out_chans=1, chans=64, num_pool_layers=4, drop_prob=0.0)
model = model.to(device)
#model = torch.nn.DataParallel(model).to(device)

##########################
# optimizer = torch.optim.Adam(model.parameters(),lr=0.001, betas=(0.9, 0.999), 
#                              eps=1e-08, weight_decay=0.0, amsgrad=False)
optimizer = torch.optim.Adam(model.parameters(),lr=0.0001)
# optimizer = torch.optim.RMSprop(model.parameters(),lr=0.0001,weight_decay=0.0)
#scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-8)

for iteration in range(TRAINING_EPOCH):
    print('Iteration:', iteration+1)
    # training
    train_loss = 0.0
    train_loss = train(model, train_dataloader, optimizer)
    writer.add_scalar("Training NMSE", train_loss, iteration+1)
    # val

    # validate each epoch
    model_copy = copy.deepcopy(model)
    model_copy = tuning(model_copy, in_tuning_dataloader, epoch = 5)
    validation_loss = evaluate(model_copy, in_validation_dataloader)
    print('In-distribution Validation NMSE: ', validation_loss.item())
    writer.add_scalar("In-distribution Validation NMSE", validation_loss.item(), iteration+1)  

    model_copy = copy.deepcopy(model)
    model_copy = tuning(model_copy, out_tuning_dataloader, epoch = 5)
    validation_loss = evaluate(model_copy, out_validation_dataloader)
    print('Out-distribution Validation NMSE: ', validation_loss.item())
    writer.add_scalar("Out-distribution Validation NMSE", validation_loss.item(), iteration+1) 
    
    #scheduler.step(val_loss_history[-1])

    # save checkpoint each epoch
    save_path_epoch = experiment_path + experiment_name + '_E' + str((iteration+1)) + '.pth'
    torch.save((model.state_dict()), save_path_epoch)
    print('Model saved to', save_path_epoch)
