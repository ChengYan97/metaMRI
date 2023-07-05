import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import tqdm
import random
import numpy as np
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
experiment_name = 'E2.1_maml_distributionTi_dataset3x500_50epoch'

# tensorboard dir
experiment_path = '/cheng/metaMRI/metaMRI/save/' + experiment_name + '/'
writer = SummaryWriter(experiment_path)

########################### hyperparametes ###########################
OUTER_EPOCH = 50
INNER_EPOCH = 2
K = 4      # K examples for inner loop training
adapt_steps = 2
adapt_lr = 0.0001   # adapt θ': α
meta_lr = 0.0001    # update real model θ: β

########################### datas  ###########################
num_train_subset = 500
num_val_subset = 200
train_path_PD = '/cheng/metaMRI/metaMRI/data_dict/knee_train_CORPD.yaml'
train_path_PDFS = '/cheng/metaMRI/metaMRI/data_dict/knee_train_CORPDFS.yaml'
train_path_FLAIR = '/cheng/metaMRI/metaMRI/data_dict/brain_train_AXFLAIR.yaml'
val_path_PD = '/cheng/metaMRI/metaMRI/data_dict/knee_val_CORPD.yaml'
val_path_PDFS = '/cheng/metaMRI/metaMRI/data_dict/knee_val_CORPDFS.yaml'
val_path_FLAIR = '/cheng/metaMRI/metaMRI/data_dict/brain_val_AXFLAIR.yaml'

# mask function and data transform
mask_function = create_mask_for_mask_type(mask_type_str = 'random', self_sup = False, 
                    center_fraction = 0.08, acceleration = 4.0, acceleration_total = 3.0)

data_transform_train = UnetDataTransform('multicoil', mask_func = mask_function, use_seed = False)
data_transform_test = UnetDataTransform('multicoil', mask_func = mask_function, use_seed=True)

# dataset
trainset_kneePD =  SliceDataset(dataset = train_path_PD, path_to_dataset='', path_to_sensmaps=None, provide_senmaps=False, 
                challenge="multicoil", transform=data_transform_train, use_dataset_cache=True, num_samples= num_train_subset)
trainset_kneePDFS =  SliceDataset(dataset = train_path_PDFS, path_to_dataset='', path_to_sensmaps=None, provide_senmaps=False, 
                challenge="multicoil", transform=data_transform_train, use_dataset_cache=True, num_samples= num_train_subset)
trainset_brainFLAIR =  SliceDataset(dataset = train_path_FLAIR, path_to_dataset='', path_to_sensmaps=None, provide_senmaps=False, 
                challenge="multicoil", transform=data_transform_train, use_dataset_cache=True, num_samples= num_train_subset)

val_set_kneePD =  SliceDataset(dataset = val_path_PD, path_to_dataset='', path_to_sensmaps=None, provide_senmaps=False, 
                challenge="multicoil", transform=data_transform_train, use_dataset_cache=True, num_samples= num_val_subset)
val_set_kneePDFS =  SliceDataset(dataset = val_path_PDFS, path_to_dataset='', path_to_sensmaps=None, provide_senmaps=False, 
                challenge="multicoil", transform=data_transform_train, use_dataset_cache=True, num_samples= num_val_subset)
val_set_brainFLAIR =  SliceDataset(dataset = val_path_FLAIR, path_to_dataset='', path_to_sensmaps=None, provide_senmaps=False, 
                challenge="multicoil", transform=data_transform_train, use_dataset_cache=True, num_samples= num_val_subset)
validation_set = torch.utils.data.ConcatDataset([val_set_kneePD, val_set_kneePDFS, val_set_brainFLAIR])

# test dataloader
validation_dataloader = torch.utils.data.DataLoader(dataset = validation_set, batch_size = 1, num_workers = 8, 
                    shuffle = False, generator = torch.Generator().manual_seed(1), pin_memory = False)


model = Unet(in_chans = 1,out_chans = 1,chans = 64, num_pool_layers = 4,drop_prob = 0.0)
#model = nn.DataParallel(model).to(device)
model = model.to(device)
maml = l2l.algorithms.MAML(model, lr=adapt_lr, first_order=False, allow_unused=True)


def MAML_K_sampler(dataset, K): 
    # Create a list of indices from 0 to length of dataset
    dataset_size = len(dataset)
    dataset_indices = list(range(dataset_size))
    # Shuffle the indices
    np.random.shuffle(dataset_indices)
    # sample several shots from dataset as one Task for MAML
    sample_indices = dataset_indices[0 : K]
    K_sampler = torch.utils.data.SubsetRandomSampler(sample_indices)
    K_examples_dataloader = torch.utils.data.DataLoader(dataset = dataset, shuffle=False, batch_size=1, sampler=K_sampler)
    # shots_per_task x [UnetSample]
    return K_examples_dataloader

def evaluate(model, dataloader, evaluate_loss, evaluate_loss_history):
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
    evaluate_loss_history.append(evaluate_loss.item())
    return evaluate_loss


optimizer = optim.Adam(maml.parameters(), meta_lr)
#optimizer = torch.optim.RMSprop(model.parameters(),lr=0.0001,weight_decay=0.0)
lossfn = nn.MSELoss(reduction='sum')


validation_loss_history = []

# outer loop
for out_iter in range(OUTER_EPOCH):    # time step
    print('Outer Loop Iteration:', out_iter+1)
    # for reproducible
    random.seed(out_iter)
    torch.manual_seed(out_iter)

    # here we consider every data distribution as a task
    task_list = [trainset_kneePD, trainset_kneePDFS, trainset_brainFLAIR]
    # sample one task Ti from list p(T)
    T_i = random.choice(task_list)

    # task set dataloader for updating the real model
    task_dataloader = torch.utils.data.DataLoader(dataset = T_i, batch_size = 1, num_workers = 4, 
                    shuffle = True, generator = torch.Generator().manual_seed(1), pin_memory = False)
    
    # enumerate one distribution/task
    meta_task_loss = 0.0
    for outer, task_batch in tqdm(enumerate(task_dataloader)):  
        # update real model
        model.train()

        ###### inner loop ######
        total_inner_loss = 0.0
        # inner_iter could be any value: "using multiple gradient updates is a straightforward extension"
        # inner loop: adapt
        for inner_iter in range(INNER_EPOCH):      
            #print('Inner Loop Iteration:', inner_iter+1)

            # sample k examples to do adaption on learner
            K_examples_dataloader = MAML_K_sampler(T_i, K)
            K_examples_inputs = []
            K_examples_targets = []
            for _, example in enumerate(K_examples_dataloader):
                K_examples_inputs.append(example.image)
                K_examples_targets.append(example.target)
            # tensor: [ (K), data ]
            K_examples_inputs = torch.stack(K_examples_inputs).to(device)
            K_examples_targets = torch.stack(K_examples_targets).to(device)

            # base learner
            learner = maml.clone()
            #learner = torch.nn.DataParallel(learner, device_ids=[0,1,2,3])

            # 5. Evaluate ∇θLTi(fθ) with respect to K examples
            for _ in range(adapt_steps): 
                K_examples_preds = learner(K_examples_inputs)
                adapt_loss = lossfn(K_examples_preds, K_examples_targets) / torch.sum(torch.abs(K_examples_targets)**2)
                #print('Inner loop adapt training loss: ', adapt_loss.item())
                # 6. Compute  adapted  parameters  with  gradient  descent: θ′i = θ − α∇θLTi(fθ)
                learner.adapt(adapt_loss)
            
            # update loss
            update_inputs = task_batch.image.unsqueeze(1).to(device)
            update_targets = task_batch.target.unsqueeze(1).to(device)
            update_output = learner(update_inputs)

            # NMSE
            loss = lossfn(update_output, update_targets) / torch.sum(torch.abs(update_targets)**2)
            # total loss for updating: ∑Ti∼p(T)LTi(fθ′i)
            total_inner_loss += loss


        # update θ through emunelate task loss
        # Update θ ← θ−β∇θ∑Ti∼p(T)LTi(fθ′i)
        meta_task_loss += total_inner_loss 
        optimizer.zero_grad()
        total_inner_loss.backward()
        optimizer.step()

    print("MAML task loss", meta_task_loss.item()/(outer+1))
    writer.add_scalar("MAML task loss", meta_task_loss.item()/(outer+1), out_iter+1)
    
    # test each outer epoch
    validation_loss = 0.0
    validation_loss = evaluate(model, validation_dataloader, validation_loss, validation_loss_history)
    print('Validation NMSE: ', validation_loss.item())
    writer.add_scalar("Validation NMSE", validation_loss.item(), out_iter+1)

    # save best model through validation dataset
    if validation_loss.item() < best_validation_loss:
        save_path_epoch = experiment_path + experiment_name + '_E' + str((out_iter+1)) + '_best.pth'
        torch.save((model.state_dict()), save_path_epoch)
        print('Model saved to', save_path_epoch)
        best_validation_loss = validation_loss.item()
    else: 
        pass

    # save checkpoint each 10 outer epoch
    if (out_iter+1) % 10 == 0:
        save_path_epoch = experiment_path + experiment_name + '_E' + str((out_iter+1)) + '.pth'
        torch.save((model.state_dict()), save_path_epoch)
        print('Model saved to', save_path_epoch)