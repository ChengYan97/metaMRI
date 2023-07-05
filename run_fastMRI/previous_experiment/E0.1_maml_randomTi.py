import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
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
#device_ids=list(range(torch.cuda.device_count()))

experiment_name = 'testmaml_v0.2_dataset3x500_20000epoch_metalr0.0001'

# tensorboard dir
experiment_path = '/cheng/metaMRI/metaMRI/save/' + experiment_name + '/'
writer = SummaryWriter(experiment_path)

num_sample_train_subset = 500
num_sample_test_subset = 200

def sample_3dataset(num_sample_subset, data_transform, path_PD, path_PDFS, path_FLAIR): 
    # data dataset x 3
    dataset_kneeCORPD =  SliceDataset(dataset = path_PD, path_to_dataset='', path_to_sensmaps=None, provide_senmaps=False, 
                    challenge="multicoil", transform=data_transform, use_dataset_cache=True, num_samples= num_sample_subset)
    dataset_kneeCORPDFS =  SliceDataset(dataset = path_PDFS, path_to_dataset='', path_to_sensmaps=None, provide_senmaps=False, 
                    challenge="multicoil", transform=data_transform, use_dataset_cache=True, num_samples= num_sample_subset)
    dataset_brainAXFLAIR =  SliceDataset(dataset = path_FLAIR, path_to_dataset='', path_to_sensmaps=None, provide_senmaps=False, 
                    challenge="multicoil", transform=data_transform, use_dataset_cache=True, num_samples= num_sample_subset)
    dataset = torch.utils.data.ConcatDataset([dataset_kneeCORPD, dataset_kneeCORPDFS, dataset_brainAXFLAIR])
    return dataset


# mask function and data transform
mask_function = create_mask_for_mask_type(mask_type_str = 'random', self_sup = False, 
                    center_fraction = 0.08, acceleration = 4.0, acceleration_total = 3.0)

data_transform_train = UnetDataTransform('multicoil', mask_func = mask_function, use_seed = False)
data_transform_test = UnetDataTransform('multicoil', mask_func = mask_function, use_seed=True)

# dataset: num_sample_subset x 3
trainset = sample_3dataset(num_sample_subset = num_sample_train_subset, data_transform = data_transform_train, 
                    path_PD = '/cheng/metaMRI/metaMRI/data_dict/knee_multicoil_train_CORPD.yaml', 
                    path_PDFS = '/cheng/metaMRI/metaMRI/data_dict/knee_multicoil_train_CORPDFS.yaml', 
                    path_FLAIR = '/cheng/metaMRI/metaMRI/data_dict/brain_multicoil_train_AXFLAIR.yaml')
testset = sample_3dataset(num_sample_subset = num_sample_test_subset, data_transform = data_transform_test, 
                    path_PD = '/cheng/metaMRI/metaMRI/data_dict/knee_multicoil_val_CORPD.yaml', 
                    path_PDFS = '/cheng/metaMRI/metaMRI/data_dict/knee_multicoil_val_CORPDFS.yaml', 
                    path_FLAIR = '/cheng/metaMRI/metaMRI/data_dict/brain_multicoil_val_AXFLAIR.yaml')

# dataloader: batch size 1 
train_dataloader = torch.utils.data.DataLoader(dataset = trainset, batch_size = 1, num_workers = 8, 
                    shuffle = True, generator = torch.Generator().manual_seed(1), pin_memory = True)
test_dataloader = torch.utils.data.DataLoader(dataset = testset, batch_size = 1, num_workers = 8, 
                    shuffle = False, generator = torch.Generator().manual_seed(1), pin_memory = True)


def MAML_task_sampler(trainset, shots_per_task): 
    # Create a list of indices from 0 to length of dataset
    dataset_size = len(trainset)
    dataset_indices = list(range(dataset_size))
    # Shuffle the indices
    np.random.shuffle(dataset_indices)

    # sample several shots from dataset as one Task for MAML
    sample_indices = dataset_indices[0 : shots_per_task]
    task_sampler = torch.utils.data.SubsetRandomSampler(sample_indices)
    train_loader_per_task = torch.utils.data.DataLoader(dataset=trainset, shuffle=False, batch_size=1, sampler=task_sampler)
    # shots_per_task x [UnetSample]
    return train_loader_per_task

model = Unet(in_chans = 1,out_chans = 1,chans = 64, num_pool_layers = 4,drop_prob = 0.0)
#model = nn.DataParallel(model).to(device)
model = model.to(device)
#model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[0,1,2,3])

adapt_lr=0.0001
maml = l2l.algorithms.MAML(model, lr=adapt_lr, first_order=False, allow_unused=True)

def evaluate(model, dataloader, val_loss_history):
    model.eval()
    total_val_loss = 0.0
    lossfn = torch.nn.MSELoss(reduction='sum')
    
    for iter, batch in enumerate(dataloader): 
        val_inputs = batch.image.unsqueeze(1).to(device)
        val_targets = batch.target.unsqueeze(1).to(device)

        val_output = model(val_inputs)

        # NMSE
        val_loss = lossfn(val_output, val_targets).item() / torch.sum(torch.abs(val_targets)**2)
        #L2_vals_dc.append(loss.item())
        total_val_loss += float(val_loss)
        
    validation_loss = total_val_loss / len(dataloader.dataset)
    val_loss_history.append(validation_loss)
    print('Validation Loss', validation_loss) 


SHOTS_PER_TASK = 6
OUTER_EPOCH = 20000
INNER_EPOCH = 4
adapt_steps = 1
meta_lr = 0.0001

#optimizer = optim.Adam(maml.parameters(), meta_lr)
optimizer = torch.optim.RMSprop(model.parameters(),lr = meta_lr,weight_decay=0.0)
lossfn = nn.MSELoss(reduction='sum')

meta_train_loss_history = []
val_loss_history = []

# outer loop
for out_iter in range(OUTER_EPOCH):    # number of task
    print('Iteration:', out_iter+1)
    # sample one task for MAML
    onetask_dataloader = MAML_task_sampler(trainset = trainset, shots_per_task = SHOTS_PER_TASK)
    
    oneTask_train_inputs = []
    oneTask_train_targets = []
    for num_shot, shot in enumerate(onetask_dataloader):
        oneTask_train_inputs.append(shot.image)
        oneTask_train_targets.append(shot.target)

    # tensor for each Task
    oneTask_train_inputs = torch.stack(oneTask_train_inputs)
    oneTask_train_targets = torch.stack(oneTask_train_targets)
    
    # model training
    model.train()
    meta_train_loss = 0.0
    
    # inner loop
    # inner_iter could be any value: "using multiple gradient updates is a straightforward extension"
    for inner_iter in range(INNER_EPOCH): 
        # learner
        learner = maml.clone()
        #learner = torch.nn.DataParallel(learner, device_ids=[0,1,2,3])

        # divide the data into support and query sets
        # [shot per task, 1, 320, 320] x 2 -> half for support half for query
        support_inputs = oneTask_train_inputs[::2].to(device)
        support_targets = oneTask_train_targets[::2].to(device)

        query_inputs = oneTask_train_inputs[1::2].to(device)
        query_targets = oneTask_train_targets[1::2].to(device)

        # Evaluate ∇θLTi(fθ) with respect to K examples
        for _ in range(adapt_steps): # adaptation_steps
            support_preds = learner(support_inputs)
            support_loss = lossfn(support_preds, support_targets) / torch.sum(torch.abs(support_targets)**2)
            learner.adapt(support_loss)

        # Compute  adapted  parameters  with  gradient  descent: θ′i = θ − α∇θLTi(fθ)
        query_preds = learner(query_inputs)
        query_loss = lossfn(query_preds, query_targets) / torch.sum(torch.abs(query_targets)**2)
        meta_train_loss += query_loss

    meta_train_loss = meta_train_loss / (SHOTS_PER_TASK * INNER_EPOCH / 2)      #(SHOTS_PER_TASK/2)

    optimizer.zero_grad()
    meta_train_loss.backward()
    optimizer.step()

    meta_train_loss_history.append(meta_train_loss.item())
    print('Meta Train Loss', meta_train_loss.item()) 
    writer.add_scalar("Meta Training NMSE", meta_train_loss_history[out_iter], out_iter)
    
    # val
    if (out_iter+1) % 100 == 0:
        evaluate(model, test_dataloader, val_loss_history)
        writer.add_scalar("Validation NMSE", val_loss_history[int(out_iter/100)], out_iter)
    
    # save checkpoint per 1000 outer epoch
    if (out_iter+1) % 1000 == 0:
        save_path_epoch = experiment_path + experiment_name + '_E' + str((out_iter+1)) + '.pth'
        torch.save((model.state_dict()), save_path_epoch)
        print('Model saved to', save_path_epoch)

### save model
save_path = experiment_path + experiment_name + '.pth'
torch.save((model.state_dict()), save_path)
print('Model saved to', save_path)