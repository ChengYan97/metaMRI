import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
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
experiment_name = 'E3.1_maml_T3x300_70epoch_r3'

# tensorboard dir
experiment_path = '/cheng/metaMRI/metaMRI/save/' + experiment_name + '/'
writer = SummaryWriter(experiment_path)

BREAKPOINT = True

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
########################### hyperparametes ###########################
OUTER_EPOCH = 200
INNER_EPOCH = 2
K = 4      # K examples for inner loop training
adapt_steps = 2
adapt_lr = 0.0001   # adapt θ': α
meta_lr = 0.0001    # update real model θ: β

########################### datas  ###########################
num_train_subset = 300
num_val_subset = 300
train_path_PD = '/cheng/metaMRI/metaMRI/data_dict/knee_train_CORPD.yaml'
train_path_PDFS = '/cheng/metaMRI/metaMRI/data_dict/knee_train_CORPDFS.yaml'
train_path_FLAIR = '/cheng/metaMRI/metaMRI/data_dict/brain_train_AXFLAIR.yaml'
val_path_PD = '/cheng/metaMRI/metaMRI/data_dict/knee_val_CORPD.yaml'
val_path_PDFS = '/cheng/metaMRI/metaMRI/data_dict/knee_val_CORPDFS.yaml'
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
train_set = torch.utils.data.ConcatDataset([trainset_kneePD, trainset_kneePDFS, trainset_brainFLAIR])

val_set_kneePD =  SliceDataset(dataset = val_path_PD, path_to_dataset='', path_to_sensmaps=None, provide_senmaps=False, 
                challenge="multicoil", transform=data_transform_train, use_dataset_cache=True, num_samples= num_val_subset)
val_set_kneePDFS =  SliceDataset(dataset = val_path_PDFS, path_to_dataset='', path_to_sensmaps=None, provide_senmaps=False, 
                challenge="multicoil", transform=data_transform_train, use_dataset_cache=True, num_samples= num_val_subset)
val_set_brainFLAIR =  SliceDataset(dataset = val_path_FLAIR, path_to_dataset='', path_to_sensmaps=None, provide_senmaps=False, 
                challenge="multicoil", transform=data_transform_train, use_dataset_cache=True, num_samples= num_val_subset)
validation_set = torch.utils.data.ConcatDataset([val_set_kneePD, val_set_kneePDFS, val_set_brainFLAIR])

# dataloader: shuffle = False, will shuffle latter
train_dataloader = torch.utils.data.DataLoader(dataset = train_set, batch_size = 5, num_workers = 0, 
                    shuffle = False, generator = torch.Generator().manual_seed(1), pin_memory = False)

validation_dataloader = torch.utils.data.DataLoader(dataset = validation_set, batch_size = 1, num_workers = 0, 
                    shuffle = False, generator = torch.Generator().manual_seed(1), pin_memory = False)

# model
model = Unet(in_chans = 1,out_chans = 1,chans = 64, num_pool_layers = 4,drop_prob = 0.0)
#model = nn.DataParallel(model).to(device)
model = model.to(device)
maml = l2l.algorithms.MAML(model, lr=adapt_lr, first_order=False, allow_unused=True)


def evaluate(model, dataloader, evaluate_loss):
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


optimizer = optim.Adam(maml.parameters(), meta_lr)
#optimizer = torch.optim.RMSprop(model.parameters(),lr=0.0001,weight_decay=0.0)
lossfn = nn.MSELoss(reduction='sum')


# breakpoint
start_epoch = -1
if BREAKPOINT:
    path_checkpoint = "/cheng/metaMRI/metaMRI/save/E3.1/E3.1_maml_T3x300_70epoch_r2/E3.1_maml_T3x300_70epoch_r2_E20.pth"  
    model.load_state_dict(torch.load(path_checkpoint)) 
    start_epoch = 20

########################### MAML training ###########################
###### outer loop ######
for out_iter in range(start_epoch, OUTER_EPOCH):    # time step
    print('Outer Loop Iteration:', out_iter+1)
    # for reproducible
    random.seed(out_iter)
    torch.manual_seed(out_iter)

    # convert dataLoader into list
    data_list = []
    for batch in train_dataloader:
        data_list.append(batch)
    # shuffle the collected list
    random.shuffle(data_list)

    # enumerate one distribution/task
    total_task_loss = 0.0
    for task_batch in data_list:
        # update real model
        model.train()

        ###### inner loop ######
        total_inner_loss = 0.0
        # inner_iter could be any value: "using multiple gradient updates is a straightforward extension"
        # inner loop: adapt
        for inner_iter in range(INNER_EPOCH):      
            #print('Inner Loop Iteration:', inner_iter+1)

            ### sample k examples to do adaption on learner: K = 4
            K_examples_inputs = task_batch.image[0:4].unsqueeze(1).to(device)
            K_examples_targets = task_batch.target[0:4].unsqueeze(1).to(device)

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
            update_inputs = task_batch.image[4].unsqueeze(0).unsqueeze(0).to(device)
            update_targets = task_batch.target[4].unsqueeze(0).unsqueeze(0).to(device)
            update_output = learner(update_inputs)

            # NMSE
            loss = lossfn(update_output, update_targets) / torch.sum(torch.abs(update_targets)**2)
            # total loss for updating: ∑Ti∼p(T)LTi(fθ′i)
            total_inner_loss += loss

        # update θ through emunelate task loss
        # Update θ ← θ−β∇θ∑Ti∼p(T)LTi(fθ′i)
        total_task_loss += total_inner_loss 
        optimizer.zero_grad()
        total_inner_loss.backward()
        optimizer.step()

    print("MAML Training NMSE", total_task_loss.item()/len(data_list))
    writer.add_scalar("MAML Training NMSE", total_task_loss.item()/len(data_list), out_iter+1)

    # validate each outer epoch
    validation_loss = 0.0
    validation_loss = evaluate(model, validation_dataloader, validation_loss)
    print('Validation NMSE: ', validation_loss.item())
    writer.add_scalar("Validation NMSE", validation_loss.item(), out_iter+1)  

    
    # save checkpoint each outer epoch
    save_path_epoch = experiment_path + experiment_name + '_E' + str((out_iter+1)) + '.pth'
    torch.save((model.state_dict()), save_path_epoch)
    print('Model saved to', save_path_epoch)