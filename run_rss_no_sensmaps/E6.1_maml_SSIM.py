#%%
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import random
import numpy as np
import copy
import torch
import learn2learn as l2l
from tqdm import tqdm
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
# The corase reconstruction is the rss of the zerofilled multi-coil kspaces
# after inverse FT.
from functions.data.transforms import UnetDataTransform_norm
# Import a torch.utils.data.Dataset class that takes a list of data examples, a path to those examples
# a data transform and outputs a torch dataset.
from functions.data.mri_dataset import SliceDataset
# Unet architecture as nn.Module
from functions.models.unet import Unet
# Function that returns a MaskFunc object either for generatig random or equispaced masks
from functions.data.subsample import create_mask_for_mask_type
from functions.training.losses import SSIMLoss

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

########################### experiment name ###########################
experiment_name = 'E6.1_maml_T2x300_200epoch'

# tensorboard dir
experiment_path = '/cheng/metaMRI/metaMRI/save/' + experiment_name + '/'
writer = SummaryWriter(experiment_path)

# seed
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

###########################  hyperparametes  ###########################
EPOCH = 200   
# enumalate the whole data once takes 180 outer loop
Inner_EPOCH = 1

K = 5      # K examples for inner loop training
K_update = 1
adapt_steps = 5
adapt_lr = 0.001   # adapt θ': α
meta_lr = 0.0001    # update real model θ: β

###########################  data & dataloader  ###########################
num_train_subset = 30
num_val_subset = 100
fewshot = 10

# data path
path_train1 = '/cheng/metaMRI/metaMRI/data_dict_narrowSlices/E6.1/knee_train_PD_Skyra_15-22.yaml'
path_train2 = '/cheng/metaMRI/metaMRI/data_dict_narrowSlices/E6.1/knee_train_PDFS_Aera_15-22.yaml'
#path_train3 = '/cheng/metaMRI/metaMRI/data_dict_narrowSlices/E6.1/brain_train_T1POST_TrioTim5-8.yaml'
path_val1 = '/cheng/metaMRI/metaMRI/data_dict_narrowSlices/E6.1/knee_val_PD_Skyra_15-22.yaml'
path_val2 = '/cheng/metaMRI/metaMRI/data_dict_narrowSlices/E6.1/knee_val_PDFS_Aera_15-22.yaml'
#path_val3 = '/cheng/metaMRI/metaMRI/data_dict_narrowSlices/E6.1/brain_val_T1POST_TrioTim5-8.yaml'

path_tuning1 = '/cheng/metaMRI/metaMRI/data_dict_narrowSlices/E6.1/knee_train_PD_Skyra_15-22.yaml'
path_tuning2 = '/cheng/metaMRI/metaMRI/data_dict_narrowSlices/E6.1/knee_train_PDFS_Aera_15-22.yaml'


# mask function and data transform
mask_function = create_mask_for_mask_type(mask_type_str = 'random', self_sup = False, 
                    center_fraction = 0.08, acceleration = 4.0, acceleration_total = 3.0)

data_transform_train = UnetDataTransform_norm('multicoil', mask_func = mask_function, use_seed=False)
data_transform_test = UnetDataTransform_norm('multicoil', mask_func = mask_function, use_seed=True)

# dataset
trainset_1 =  SliceDataset(dataset = path_train1, path_to_dataset='', path_to_sensmaps=None, provide_senmaps=False, 
                challenge="multicoil", transform=data_transform_train, use_dataset_cache=True, num_samples= num_train_subset)
trainset_2 =  SliceDataset(dataset = path_train2, path_to_dataset='', path_to_sensmaps=None, provide_senmaps=False, 
                challenge="multicoil", transform=data_transform_train, use_dataset_cache=True, num_samples= num_train_subset)

tuning_set1 = SliceDataset(dataset = path_tuning1, path_to_dataset='', path_to_sensmaps=None, provide_senmaps=False, 
                challenge="multicoil", transform=data_transform_train, use_dataset_cache=True, num_samples= fewshot)
tuning_set2 = SliceDataset(dataset = path_tuning2, path_to_dataset='', path_to_sensmaps=None, provide_senmaps=False, 
                challenge="multicoil", transform=data_transform_train, use_dataset_cache=True, num_samples= fewshot)

val_set1 =  SliceDataset(dataset = path_val1, path_to_dataset='', path_to_sensmaps=None, provide_senmaps=False, 
                challenge="multicoil", transform=data_transform_test, use_dataset_cache=True, num_samples= num_val_subset)
val_set2 =  SliceDataset(dataset = path_val2, path_to_dataset='', path_to_sensmaps=None, provide_senmaps=False, 
                challenge="multicoil", transform=data_transform_test, use_dataset_cache=True, num_samples= num_val_subset)

# dataloader
train_dataloader_1 = torch.utils.data.DataLoader(dataset = trainset_1, batch_size = K + K_update,
                    shuffle = True, generator = torch.Generator().manual_seed(1), pin_memory = True)
train_dataloader_2 = torch.utils.data.DataLoader(dataset = trainset_2, batch_size = K + K_update,
                    shuffle = True, generator = torch.Generator().manual_seed(1), pin_memory = True)

tuning_dataloader_1 = torch.utils.data.DataLoader(dataset = tuning_set1, batch_size = 1, 
                    shuffle = True, generator = torch.Generator().manual_seed(1), pin_memory = False)
tuning_dataloader_2 = torch.utils.data.DataLoader(dataset = tuning_set2, batch_size = 1, 
                    shuffle = True, generator = torch.Generator().manual_seed(1), pin_memory = False)

validation_dataloader_1 = torch.utils.data.DataLoader(dataset = val_set1, batch_size = 1, 
                    shuffle = False, generator = torch.Generator().manual_seed(1), pin_memory = False)
validation_dataloader_2 = torch.utils.data.DataLoader(dataset = val_set2, batch_size = 1, 
                    shuffle = False, generator = torch.Generator().manual_seed(1), pin_memory = False)



#%% Check the data 
###########################  model  ###########################
model = Unet(in_chans = 1,out_chans = 1,chans = 32, num_pool_layers = 4,drop_prob = 0.0)
#model = nn.DataParallel(model).to(device)
model = model.to(device)
maml = l2l.algorithms.MAML(model, lr=adapt_lr, first_order=False, allow_unused=True)


def tuning(model, dataloader, epoch): 
    model.train()
    for iteration in range(epoch):
        total_loss = 0.0
        for iter, batch in enumerate(dataloader): 
            input_image, target_image, mean, std, fname, slice_num = batch
            inputs = input_image.to(device)
            targets = target_image.to(device)
            std = std.to(device)
            mean = mean.to(device)

            output = model(inputs)
            output = output * std + mean
            
            # SSIM = 1 - loss
            loss = ssim_fct(output, targets, data_range = targets.max().unsqueeze(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model


def evaluate(model, dataloader):
    model.eval()
    total_loss = 0.0
    for iter, batch in enumerate(dataloader): 
        input_image, target_image, mean, std, fname, slice_num = batch
        inputs = input_image.to(device)
        targets = target_image.to(device)
        std = std.to(device)
        mean = mean.to(device)

        output = model(inputs)
        output = output * std + mean
        # SSIM = 1 - loss
        loss = ssim_fct(output, targets, data_range = targets.max().unsqueeze(0))
        total_loss += (1-loss.item())
    evaluate_loss = total_loss / len(dataloader)
    return evaluate_loss


###########################  MAML training  ###########################
optimizer = optim.Adam(maml.parameters(), meta_lr)
ssim_fct = SSIMLoss()


### one training loop include 180 outer loop
for iter_ in range(EPOCH):    
    print('Iteration:', iter_+1)

    # update real model
    model.train()

    # shuffle the data and generate iterator
    iterator_train_1 = iter(train_dataloader_1)
    iterator_train_2 = iter(train_dataloader_2)
    iterator_list = [iterator_train_1, iterator_train_2]

    # generate a list for random sample specific dataloader
    sample_list = [0]*len(train_dataloader_1) + [1]*len(train_dataloader_2)
    random.shuffle(sample_list)

    # enumerate one distribution/task

    # use the list to get the random train dataloader

    ###### 2: outer loop ######
    # here we consider 180 outer loop as one training loop
    meta_training_loss = 0.0
    ###### 3. Sample batch of tasks Ti ~ p(T) ######
    # sample 2 batch at one time
    for index in tqdm(range(0, len(sample_list), Inner_EPOCH)):   
        print('Outer loop: ', int((index/Inner_EPOCH)+1))
        # index = 0, 2, 4, ...
        i = sample_list[index : index+Inner_EPOCH]
        # i = [ , ]
        total_update_loss = 0.0
        ###### 4: inner loop ######
        # Ti only contain one task: (K+K_update) data
        for inner_iter in range(Inner_EPOCH):
            print('Inner loop: ', inner_iter+1)
            # load one batch from random dataloader
            iterator = iterator_list[i[inner_iter]]  # i = [ , ], 1st loop i[0], 2nd loop i[1]
            task_batch = next(iterator)

            # load K + K_update data
            input_image, target_image, mean, std, fname, slice_num = task_batch

            ### sample k examples to do adaption on learner: K = 4
            K_examples_inputs = input_image[0:K].to(device)
            K_examples_targets = target_image[0:K].to(device)
            K_mean = mean[0:K].to(device)
            K_std = std[0:K].to(device)

            # base learner
            learner = maml.clone()      #learner = torch.nn.DataParallel(learner, device_ids=[0,1,2,3])

            # adapt learner several step to K examples
            for _ in range(adapt_steps): 
                ###### 5. Evaluate ∇θLTi(fθ) with respect to K examples ######
                K_examples_preds = learner(K_examples_inputs)
                K_examples_preds = K_examples_preds * K_std + K_mean
                adapt_loss = ssim_fct(K_examples_preds, K_examples_targets, data_range = K_examples_targets.max().unsqueeze(0))
                ###### 6. Compute  adapted  parameters  with  gradient  descent: θ′i = θ − α∇θLTi(fθ) ######
                learner.adapt(adapt_loss)
            
            # sample the K_update example for updating the model
            update_inputs = input_image[K:(K+K_update)].to(device)
            update_targets = target_image[K:(K+K_update)].to(device)
            update_mean = mean[K:(K+K_update)].to(device)
            update_std = std[K:(K+K_update)].to(device)

            ###### 7: inner loop end ######
            # for calculation efficient, some loss are still cumputed in this loop

            ####### 8. Update θ ← θ−β∇θ∑Ti∼p(T)LTi(fθ′i)   ######   
            # LTi(fθ′i)
            update_output = learner(update_inputs)
            update_output = update_output * update_std + update_mean
            update_loss = ssim_fct(update_output, update_targets, data_range = update_targets.max().unsqueeze(0))
            # ∑Ti∼p(T)LTi(fθ′i): Ti only contain one task
            total_update_loss += update_loss

        # del task_batch  # avoid cpu memory leak
        # del learner     # gpu

        # Update θ ← θ−β∇θ∑Ti∼p(T)LTi(fθ′i)
        optimizer.zero_grad()
        total_update_loss.backward()
        optimizer.step()
        # total_update_loss should be the meta training loss, 
        # but we want some value can evaluate the training through whole dataset
        # we use the mean of 180 outer loop loss as the meta training loss
        meta_training_loss += total_update_loss.item()

    print("Meta Training SSIM (MAML)", (1-meta_training_loss/len(sample_list)))
    writer.add_scalar("Meta Training SSIM (MAML)", (1-meta_training_loss/len(sample_list)), iter_+1)

    # validate each epoch
    model_copy = copy.deepcopy(model)
    model_copy = tuning(model_copy, tuning_dataloader_1, epoch = 5)
    validation_loss1 = evaluate(model_copy, validation_dataloader_1)
    model_copy = copy.deepcopy(model)
    model_copy = tuning(model_copy, tuning_dataloader_2, epoch = 5)
    validation_loss2 = evaluate(model_copy, validation_dataloader_2)
    validation_loss = (validation_loss1+validation_loss2)/2
    print('Validation SSIM: ', validation_loss)
    writer.add_scalar("Validation SSIM", validation_loss, iter_+1)  

    # save checkpoint each outer epoch
    save_path_epoch = experiment_path + experiment_name + '_E' + str((iter_+1)) + '.pth'
    torch.save((model.state_dict()), save_path_epoch)
    print('Model saved to', save_path_epoch)