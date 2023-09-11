#%%
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
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
experiment_name = "E10.2_maml(NMSE-lre-3)_T8x200_250epoch"

# tensorboard dir
experiment_path = '/cheng/metaMRI/metaMRI/save/' + experiment_name + '/'
writer = SummaryWriter(experiment_path)

# seed
SEED = 1
random.seed(SEED)
np.random.seed(SEED)
torch.cuda.manual_seed(SEED)
torch.manual_seed(SEED)

###########################  hyperparametes  ###########################
EPOCH = 250   
# enumalate the whole data once takes 180 outer loop
Inner_EPOCH = 1

K = 4      # K examples for inner loop training
K_update = 1
adapt_steps = 5
adapt_lr = 0.0001   # adapt θ': α
meta_lr = 0.001    # update real model θ: β

###########################  data & dataloader  ###########################
num_train_subset = 200
num_val_subset = 100
fewshot = 10

# data path
path_train1 = '/cheng/metaMRI/metaMRI/data_dict/E10.2/P/knee_train_PD_Aera_2-9.yaml'
path_train2 = '/cheng/metaMRI/metaMRI/data_dict/E10.2/P/knee_train_PD_Aera_15-22.yaml'
path_train3 = '/cheng/metaMRI/metaMRI/data_dict/E10.2/P/knee_train_PD_Biograph_15-22.yaml'
path_train4 = '/cheng/metaMRI/metaMRI/data_dict/E10.2/P/brain_train_T1POST_TrioTim_5-8.yaml'
path_train5 = '/cheng/metaMRI/metaMRI/data_dict/E10.2/P/knee_train_PDFS_Aera_2-9.yaml'
path_train6 = '/cheng/metaMRI/metaMRI/data_dict/E10.2/P/knee_train_PDFS_Aera_15-22.yaml'
path_train7 = '/cheng/metaMRI/metaMRI/data_dict/E10.2/P/knee_train_PDFS_Biograph_15-22.yaml'
path_train8 = '/cheng/metaMRI/metaMRI/data_dict/E10.2/P/brain_train_FLAIR_Skyra_5-8.yaml'

path_val1 = '/cheng/metaMRI/metaMRI/data_dict/E10.2/P/knee_val_PD_Aera_2-9.yaml'
path_val2 = '/cheng/metaMRI/metaMRI/data_dict/E10.2/P/knee_val_PD_Aera_15-22.yaml'
path_val3 = '/cheng/metaMRI/metaMRI/data_dict/E10.2/P/knee_val_PD_Biograph_15-22.yaml'
path_val4 = '/cheng/metaMRI/metaMRI/data_dict/E10.2/P/brain_val_T1POST_TrioTim_5-8.yaml'
path_val5 = '/cheng/metaMRI/metaMRI/data_dict/E10.2/P/knee_val_PDFS_Aera_2-9.yaml'
path_val6 = '/cheng/metaMRI/metaMRI/data_dict/E10.2/P/knee_val_PDFS_Aera_15-22.yaml'
path_val7 = '/cheng/metaMRI/metaMRI/data_dict/E10.2/P/knee_val_PDFS_Biograph_15-22.yaml'
path_val8 = '/cheng/metaMRI/metaMRI/data_dict/E10.2/P/brain_val_FLAIR_Skyra_5-8.yaml'


# mask function and data transform
mask_function = create_mask_for_mask_type(mask_type_str = 'random', self_sup = False, 
                    center_fraction = 0.08, acceleration = 4.0, acceleration_total = 3.0)

data_transform_train = UnetDataTransform_norm('multicoil', mask_func = mask_function, use_seed=False, mode='train')
data_transform = UnetDataTransform_norm('multicoil', mask_func = mask_function, use_seed=True, mode='adapt')

# training dataset and data loader
path_train_list = [path_train1, path_train2, path_train3, path_train4, path_train5, path_train6, path_train7, path_train8]
train_dataloader_list = []
for path_train in path_train_list:
    trainset = SliceDataset(dataset = path_train, path_to_dataset='', path_to_sensmaps=None, provide_senmaps=False, 
                challenge="multicoil", transform=data_transform_train, use_dataset_cache=True, num_samples= num_train_subset)
    train_dataloader = torch.utils.data.DataLoader(dataset = trainset, batch_size = K + K_update,
                    shuffle = True, generator = torch.Generator().manual_seed(1), pin_memory = True)
    train_dataloader_list.append(train_dataloader)


# adaptation dataset and data loader
adapt_dataloader_list = []
for path_adapt in path_train_list:
    adapt_set = SliceDataset(dataset = path_adapt, path_to_dataset='', path_to_sensmaps=None, provide_senmaps=False, 
                challenge="multicoil", transform=data_transform, use_dataset_cache=True, num_samples= fewshot)
    adapt_dataloader = torch.utils.data.DataLoader(dataset = adapt_set, batch_size = 1, 
                    shuffle = False, generator = torch.Generator().manual_seed(1), pin_memory = False)
    adapt_dataloader_list.append(adapt_dataloader)

# validation dataset and data loader
path_val_list = [path_val1, path_val2, path_val3, path_val4, path_val5, path_val6, path_val7, path_val8]
validation_dataloader_list = []
for path_val in path_val_list:
    val_set = SliceDataset(dataset = path_val, path_to_dataset='', path_to_sensmaps=None, provide_senmaps=False, 
                challenge="multicoil", transform=data_transform, use_dataset_cache=True, num_samples= num_val_subset)
    validation_dataloader = torch.utils.data.DataLoader(dataset = val_set, batch_size = 1, 
                    shuffle = False, generator = torch.Generator().manual_seed(1), pin_memory = False)
    validation_dataloader_list.append(validation_dataloader)


#%% Check the data 
###########################  model  ###########################
model = Unet(in_chans = 1,out_chans = 1,chans = 32, num_pool_layers = 4,drop_prob = 0.0)
#model = nn.DataParallel(model).to(device)
model = model.to(device)
maml = l2l.algorithms.MAML(model, lr=adapt_lr, first_order=False, allow_unused=True)


def tuning(model, dataloader, epoch): 
    model.train()
    tuning_optimizer = torch.optim.Adam(model.parameters(), 0.0001)
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

            loss = lossfn(output, targets) / torch.sum(torch.abs(targets)**2)

            tuning_optimizer.zero_grad()
            loss.backward()
            tuning_optimizer.step()
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

        loss = lossfn(output, targets) / torch.sum(torch.abs(targets)**2)
        total_loss += loss.item()
    evaluate_loss = total_loss / len(dataloader)
    return evaluate_loss


###########################  MAML training  ###########################
optimizer = optim.Adam(maml.parameters(), meta_lr)
lossfn = nn.MSELoss(reduction='sum')

### one training loop include 180 outer loop
for iter_ in range(EPOCH):    
    print('Iteration:', iter_+1)

    # update real model
    model.train()

    # shuffle the data and generate iterator
    iterator_list = [iter(data_loader) for data_loader in train_dataloader_list]

    # generate a list for random sample specific dataloader
    batch_num = int(num_train_subset/(K+K_update))
    sample_list = [0]*batch_num+[1]*batch_num+[2]*batch_num+[3]*batch_num+[4]*batch_num+[5]*batch_num+[6]*batch_num+[7]*batch_num
    random.shuffle(sample_list)

    # enumerate one distribution/task

    # use the list to get the random train dataloader

    ###### 2: outer loop ######
    # here we consider 180 outer loop as one training loop
    meta_training_loss = 0.0
    meta_adaptation_loss = 0.0
    ###### 3. Sample batch of tasks Ti ~ p(T) ######
    # sample 2 batch at one time
    for index in tqdm(range(0, len(sample_list), Inner_EPOCH)):   
        print('Outer loop: ', int((index/Inner_EPOCH)+1))
        # index = 0, 2, 4, ...
        i = sample_list[index : index+Inner_EPOCH]
        # i = [ , ]
        total_update_loss = 0.0
        total_adapt_loss = 0.0
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
                adapt_loss = lossfn(K_examples_preds, K_examples_targets) / torch.sum(torch.abs(K_examples_targets)**2)
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
            update_loss = lossfn(update_output, update_targets) / torch.sum(torch.abs(update_targets)**2)
            # ∑Ti∼p(T)LTi(fθ′i): Ti only contain one task
            total_update_loss += update_loss
            total_adapt_loss += adapt_loss.item()

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
        meta_adaptation_loss += total_adapt_loss

    print("Meta Adaptation NMSE (MAML)", meta_adaptation_loss/len(sample_list))
    writer.add_scalar("Meta Adaptation NMSE (MAML)", meta_adaptation_loss/len(sample_list), iter_+1)

    print("Meta Training NMSE (MAML)", meta_training_loss/len(sample_list))
    writer.add_scalar("Meta Training NMSE (MAML)", meta_training_loss/len(sample_list), iter_+1)
    
    # validate each epoch
    total_validation_loss = 0.0
    for val_iter in range(8): 
        model_copy = copy.deepcopy(model)
        model_copy = tuning(model_copy, adapt_dataloader_list[val_iter], epoch = 5)
        val_loss = evaluate(model_copy, validation_dataloader_list[val_iter])
        total_validation_loss += val_loss

    validation_loss = total_validation_loss/8
    print('Validation NMSE: ', validation_loss)
    writer.add_scalar("Validation NMSE", validation_loss, iter_+1)  

    # save checkpoint each outer epoch
    save_path_epoch = experiment_path + experiment_name + '_E' + str((iter_+1)) + '.pth'
    torch.save((model.state_dict()), save_path_epoch)
    print('Model saved to', save_path_epoch)