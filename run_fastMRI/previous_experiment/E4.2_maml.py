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
experiment_name = 'E4.2_maml_T3x300_150epoch'

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

K = 4      # K examples for inner loop training
K_update = 1
adapt_steps = 4
adapt_lr = 0.0001   # adapt θ': α
meta_lr = 0.0001    # update real model θ: β

###########################  data & dataloader  ###########################
num_train_subset = 300
num_val_subset = 300
train_path_PD = '/cheng/metaMRI/metaMRI/data_dict/knee_train_CORPD.yaml'
train_path_PDFS = '/cheng/metaMRI/metaMRI/data_dict/knee_train_CORPDFS.yaml'
train_path_FLAIR = '/cheng/metaMRI/metaMRI/data_dict/brain_train_AXFLAIR.yaml'
val_path_FLAIR = '/cheng/metaMRI/metaMRI/data_dict/brain_val_AXFLAIR.yaml'

# mask function and data transform
mask_function = create_mask_for_mask_type(mask_type_str = 'random', self_sup = False, 
                    center_fraction = 0.08, acceleration = 4.0, acceleration_total = 3.0)

data_transform_train = UnetDataTransform('multicoil', mask_func = mask_function, use_seed=False)
data_transform_test = UnetDataTransform('multicoil', mask_func = mask_function, use_seed=True)

# dataset
trainset_kneePD =  SliceDataset(dataset = train_path_PD, path_to_dataset='', path_to_sensmaps=None, provide_senmaps=False, 
                challenge="multicoil", transform=data_transform_train, use_dataset_cache=True, num_samples= num_train_subset)
trainset_kneePDFS =  SliceDataset(dataset = train_path_PDFS, path_to_dataset='', path_to_sensmaps=None, provide_senmaps=False, 
                challenge="multicoil", transform=data_transform_train, use_dataset_cache=True, num_samples= num_train_subset)
trainset_brainFLAIR =  SliceDataset(dataset = train_path_FLAIR, path_to_dataset='', path_to_sensmaps=None, provide_senmaps=False, 
                challenge="multicoil", transform=data_transform_train, use_dataset_cache=True, num_samples= num_train_subset)
tuning_brainFLAIR = SliceDataset(dataset = train_path_FLAIR, path_to_dataset='', path_to_sensmaps=None, provide_senmaps=False, 
                challenge="multicoil", transform=data_transform_train, use_dataset_cache=True, num_samples= 10)
val_set_brainFLAIR =  SliceDataset(dataset = val_path_FLAIR, path_to_dataset='', path_to_sensmaps=None, provide_senmaps=False, 
                challenge="multicoil", transform=data_transform_train, use_dataset_cache=True, num_samples= num_val_subset)
# dataloader
train_dataloader_kneePD = torch.utils.data.DataLoader(dataset = trainset_kneePD, batch_size = K + K_update, num_workers = 0, 
                    shuffle = True, generator = torch.Generator().manual_seed(1), pin_memory = True)
train_dataloader_kneePDFS = torch.utils.data.DataLoader(dataset = trainset_kneePDFS, batch_size = K + K_update, num_workers = 0, 
                    shuffle = True, generator = torch.Generator().manual_seed(1), pin_memory = True)
train_dataloader_brainFLAIR = torch.utils.data.DataLoader(dataset = trainset_brainFLAIR, batch_size = K + K_update, num_workers = 0, 
                    shuffle = True, generator = torch.Generator().manual_seed(1), pin_memory = True)
tuning_dataloader_brainFLAIR = torch.utils.data.DataLoader(dataset = tuning_brainFLAIR, batch_size = 1, num_workers = 0, 
                    shuffle = True, generator = torch.Generator().manual_seed(1), pin_memory = False)
validation_dataloader_brainFLAIR = torch.utils.data.DataLoader(dataset = val_set_brainFLAIR, batch_size = 1, num_workers = 0, 
                    shuffle = False, generator = torch.Generator().manual_seed(1), pin_memory = False)


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
    iterator_train_kneePD = iter(train_dataloader_kneePD)
    iterator_train_kneePDFS = iter(train_dataloader_kneePDFS)
    iterator_train_brainFLAIR = iter(train_dataloader_brainFLAIR)
    iterator_list = [iterator_train_kneePD, iterator_train_kneePDFS, iterator_train_brainFLAIR]

    # generate a list for random sample specific dataloader
    sample_list = [0]*len(train_dataloader_kneePD) + [1]*len(train_dataloader_kneePDFS) + [2]*len(train_dataloader_brainFLAIR)
    random.shuffle(sample_list)

    # enumerate one distribution/task

    # use the list to get the random train dataloader

    ###### 2: outer loop ######
    # here we consider 180 outer loop as one training loop
    meta_training_loss = 0.0
    ###### 3. Sample batch of tasks Ti ~ p(T) ######
    for i in tqdm(sample_list): 
        # load one batch from random dataloader
        iterator = iterator_list[i]
        task_batch = next(iterator)
        
        total_update_loss = 0.0
        ###### 4: inner loop ######
        # Ti only contain one task: (K+K_update) data
        for inner_iter in range(Inner_EPOCH):
            ### sample k examples to do adaption on learner: K = 4
            K_examples_inputs = task_batch.image[0:4].unsqueeze(1).to(device)
            K_examples_targets = task_batch.target[0:4].unsqueeze(1).to(device)

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
            update_inputs = task_batch.image[4].unsqueeze(0).unsqueeze(0).to(device)
            update_targets = task_batch.target[4].unsqueeze(0).unsqueeze(0).to(device)

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
    model_copy = tuning(model_copy, tuning_dataloader_brainFLAIR, epoch = 5)
    validation_loss = evaluate(model_copy, validation_dataloader_brainFLAIR)
    print('Validation NMSE: ', validation_loss.item())
    writer.add_scalar("Validation NMSE", validation_loss.item(), iter_+1)  


    # save checkpoint each outer epoch
    save_path_epoch = experiment_path + experiment_name + '_E' + str((iter_+1)) + '.pth'
    torch.save((model.state_dict()), save_path_epoch)
    print('Model saved to', save_path_epoch)