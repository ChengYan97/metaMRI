#%%
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
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
from functions.data.transforms import UnetDataTransform
# Import a torch.utils.data.Dataset class that takes a list of data examples, a path to those examples
# a data transform and outputs a torch dataset.
from functions.data.mri_dataset import SliceDataset
# Unet architecture as nn.Module
from functions.models.unet import Unet
# Function that returns a MaskFunc object either for generatig random or equispaced masks
from functions.data.subsample import create_mask_for_mask_type

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

########################### experiment name ###########################
experiment_name = 'E5.1_maml_T9x100_200epoch'

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
Inner_EPOCH = 2

K = 4      # K examples for inner loop training
K_update = 1
adapt_steps = 2
adapt_lr = 0.0001   # adapt θ': α
meta_lr = 0.0001    # update real model θ: β

###########################  data & dataloader  ###########################
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

in_tuning_set = SliceDataset(dataset = in_tuning_path, path_to_dataset='', path_to_sensmaps=None, provide_senmaps=False, 
                challenge="multicoil", transform=data_transform_train, use_dataset_cache=True, num_samples= fewshot)
out_tuning_set = SliceDataset(dataset =  out_tuning_path, path_to_dataset='', path_to_sensmaps=None, provide_senmaps=False, 
                challenge="multicoil", transform=data_transform_train, use_dataset_cache=True, num_samples= fewshot)


in_val_set =  SliceDataset(dataset = in_val_path, path_to_dataset='', path_to_sensmaps=None, provide_senmaps=False, 
                challenge="multicoil", transform=data_transform_train, use_dataset_cache=True, num_samples= num_val_subset)
out_val_set =  SliceDataset(dataset = out_val_path, path_to_dataset='', path_to_sensmaps=None, provide_senmaps=False, 
                challenge="multicoil", transform=data_transform_train, use_dataset_cache=True, num_samples= num_val_subset)


# dataloader
train_dataloader_1 = torch.utils.data.DataLoader(dataset = trainset_1, batch_size = K + K_update, num_workers = 0, 
                    shuffle = True, generator = torch.Generator().manual_seed(1), pin_memory = True)
train_dataloader_2 = torch.utils.data.DataLoader(dataset = trainset_2, batch_size = K + K_update, num_workers = 0, 
                    shuffle = True, generator = torch.Generator().manual_seed(1), pin_memory = True)
train_dataloader_3 = torch.utils.data.DataLoader(dataset = trainset_3, batch_size = K + K_update, num_workers = 0, 
                    shuffle = True, generator = torch.Generator().manual_seed(1), pin_memory = True)
train_dataloader_4 = torch.utils.data.DataLoader(dataset = trainset_4, batch_size = K + K_update, num_workers = 0, 
                    shuffle = True, generator = torch.Generator().manual_seed(1), pin_memory = True)
train_dataloader_5 = torch.utils.data.DataLoader(dataset = trainset_5, batch_size = K + K_update, num_workers = 0, 
                    shuffle = True, generator = torch.Generator().manual_seed(1), pin_memory = True)
train_dataloader_6 = torch.utils.data.DataLoader(dataset = trainset_6, batch_size = K + K_update, num_workers = 0, 
                    shuffle = True, generator = torch.Generator().manual_seed(1), pin_memory = True)
train_dataloader_7 = torch.utils.data.DataLoader(dataset = trainset_7, batch_size = K + K_update, num_workers = 0, 
                    shuffle = True, generator = torch.Generator().manual_seed(1), pin_memory = True)
train_dataloader_8 = torch.utils.data.DataLoader(dataset = trainset_8, batch_size = K + K_update, num_workers = 0, 
                    shuffle = True, generator = torch.Generator().manual_seed(1), pin_memory = True)
train_dataloader_9 = torch.utils.data.DataLoader(dataset = trainset_9, batch_size = K + K_update, num_workers = 0, 
                    shuffle = True, generator = torch.Generator().manual_seed(1), pin_memory = True)

in_tuning_dataloader = torch.utils.data.DataLoader(dataset = in_tuning_set, batch_size = 1, num_workers = 0, 
                    shuffle = True, generator = torch.Generator().manual_seed(1), pin_memory = False)
out_tuning_dataloader = torch.utils.data.DataLoader(dataset = out_tuning_set, batch_size = 1, num_workers = 0, 
                    shuffle = True, generator = torch.Generator().manual_seed(1), pin_memory = False)
in_validation_dataloader = torch.utils.data.DataLoader(dataset = in_val_set, batch_size = 1, num_workers = 0, 
                    shuffle = False, generator = torch.Generator().manual_seed(1), pin_memory = False)
out_validation_dataloader = torch.utils.data.DataLoader(dataset = out_val_set, batch_size = 1, num_workers = 0, 
                    shuffle = False, generator = torch.Generator().manual_seed(1), pin_memory = False)


#%% Check the data 
###########################  model  ###########################
model = Unet(in_chans = 1,out_chans = 1,chans = 64, num_pool_layers = 4,drop_prob = 0.0)
#model = nn.DataParallel(model).to(device)
model = model.to(device)
maml = l2l.algorithms.MAML(model, lr=adapt_lr, first_order=False, allow_unused=True)


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


###########################  MAML training  ###########################
optimizer = optim.Adam(maml.parameters(), meta_lr)
#optimizer = torch.optim.RMSprop(model.parameters(),lr=0.0001,weight_decay=0.0)
lossfn = nn.MSELoss(reduction='sum')


### one training loop include 180 outer loop
for iter_ in range(EPOCH):    
    print('Iteration:', iter_+1)

    # update real model
    model.train()

    # shuffle the data and generate iterator
    iterator_train_1 = iter(train_dataloader_1)
    iterator_train_2 = iter(train_dataloader_2)
    iterator_train_3 = iter(train_dataloader_3)
    iterator_train_4 = iter(train_dataloader_4)
    iterator_train_5 = iter(train_dataloader_5)
    iterator_train_6 = iter(train_dataloader_6)
    iterator_train_7 = iter(train_dataloader_7)
    iterator_train_8 = iter(train_dataloader_8)
    iterator_train_9 = iter(train_dataloader_9)
    iterator_list = [iterator_train_1, iterator_train_2, iterator_train_3,
                    iterator_train_4, iterator_train_5, iterator_train_6,
                    iterator_train_7, iterator_train_8, iterator_train_9]

    # generate a list for random sample specific dataloader
    sample_list = [0]*len(train_dataloader_1) + [1]*len(train_dataloader_2) + [2]*len(train_dataloader_3) \
                + [3]*len(train_dataloader_4) + [4]*len(train_dataloader_5) + [5]*len(train_dataloader_6) \
                + [6]*len(train_dataloader_7) + [7]*len(train_dataloader_8) + [8]*len(train_dataloader_9)
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

            ### sample k examples to do adaption on learner: K = 4
            K_examples_inputs = task_batch.image[0:K].unsqueeze(1).to(device)
            K_examples_targets = task_batch.target[0:K].unsqueeze(1).to(device)

            # base learner
            learner = maml.clone()      #learner = torch.nn.DataParallel(learner, device_ids=[0,1,2,3])

            # adapt learner several step to K examples
            for _ in range(adapt_steps): 
                ###### 5. Evaluate ∇θLTi(fθ) with respect to K examples ######
                K_examples_preds = learner(K_examples_inputs)
                adapt_loss = lossfn(K_examples_preds, K_examples_targets) / torch.sum(torch.abs(K_examples_targets)**2)
                ###### 6. Compute  adapted  parameters  with  gradient  descent: θ′i = θ − α∇θLTi(fθ) ######
                learner.adapt(adapt_loss)
            
            # sample the K_update example for updating the model
            update_inputs = task_batch.image[K:(K+K_update)].unsqueeze(1).to(device)
            update_targets = task_batch.target[K:(K+K_update)].unsqueeze(1).to(device)

            ###### 7: inner loop end ######
            # for calculation efficient, some loss are still cumputed in this loop

            ####### 8. Update θ ← θ−β∇θ∑Ti∼p(T)LTi(fθ′i)   ######   
            # LTi(fθ′i)
            update_output = learner(update_inputs)
            update_loss = lossfn(update_output, update_targets) / torch.sum(torch.abs(update_targets)**2)
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

    print("Meta Training NMSE (MAML)", meta_training_loss/len(sample_list))
    writer.add_scalar("Meta Training NMSE (MAML)", meta_training_loss/len(sample_list), iter_+1)

    # validate each epoch
    model_copy = copy.deepcopy(model)
    model_copy = tuning(model_copy, in_tuning_dataloader, epoch = 5)
    validation_loss = evaluate(model_copy, in_validation_dataloader)
    print('In-distribution Validation NMSE: ', validation_loss.item())
    writer.add_scalar("In-distribution Validation NMSE", validation_loss.item(), iter_+1)  

    model_copy = copy.deepcopy(model)
    model_copy = tuning(model_copy, out_tuning_dataloader, epoch = 5)
    validation_loss = evaluate(model_copy, out_validation_dataloader)
    print('Out-distribution Validation NMSE: ', validation_loss.item())
    writer.add_scalar("Out-distribution Validation NMSE", validation_loss.item(), iter_+1) 


    # save checkpoint each outer epoch
    save_path_epoch = experiment_path + experiment_name + '_E' + str((iter_+1)) + '.pth'
    torch.save((model.state_dict()), save_path_epoch)
    print('Model saved to', save_path_epoch)