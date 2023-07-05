import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
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

experiment_name = 'maml_3-dataset500_20000epoch'

# tensorboard dir
experiment_path = '/cheng/metaMRI/metaMRI/save/' + experiment_name + '/'
writer = SummaryWriter(experiment_path)

# hyperparametes
num_train_subset = 500
num_test_subset = 200
K = 4      # K examples for inner loop training
OUTER_EPOCH = 20000
INNER_EPOCH = 2
adapt_steps = 1
adapt_lr = 0.0001   # update θ': α
meta_lr = 0.0001    # update real model θ: β
train_val_split = [497,3]


# mask function and data transform
mask_function = create_mask_for_mask_type(mask_type_str = 'random', self_sup = False, 
                    center_fraction = 0.08, acceleration = 4.0, acceleration_total = 3.0)

data_transform_train = UnetDataTransform('multicoil', mask_func = mask_function, use_seed = False)
data_transform_test = UnetDataTransform('multicoil', mask_func = mask_function, use_seed=True)

# datas path
train_path_PD = '/cheng/metaMRI/metaMRI/data_dict/knee_multicoil_train_CORPD.yaml'
train_path_PDFS = '/cheng/metaMRI/metaMRI/data_dict/knee_multicoil_train_CORPDFS.yaml'
train_path_FLAIR = '/cheng/metaMRI/metaMRI/data_dict/brain_multicoil_train_AXFLAIR.yaml'
test_path_PD = '/cheng/metaMRI/metaMRI/data_dict/knee_multicoil_val_CORPD.yaml'
test_path_PDFS = '/cheng/metaMRI/metaMRI/data_dict/knee_multicoil_val_CORPDFS.yaml'
test_path_FLAIR = '/cheng/metaMRI/metaMRI/data_dict/brain_multicoil_val_AXFLAIR.yaml'

# dataset
trainset_kneePD =  SliceDataset(dataset = train_path_PD, path_to_dataset='', path_to_sensmaps=None, provide_senmaps=False, 
                challenge="multicoil", transform=data_transform_train, use_dataset_cache=True, num_samples= num_train_subset)
trainset_kneePDFS =  SliceDataset(dataset = train_path_PDFS, path_to_dataset='', path_to_sensmaps=None, provide_senmaps=False, 
                challenge="multicoil", transform=data_transform_train, use_dataset_cache=True, num_samples= num_train_subset)
trainset_brainFLAIR =  SliceDataset(dataset = train_path_FLAIR, path_to_dataset='', path_to_sensmaps=None, provide_senmaps=False, 
                challenge="multicoil", transform=data_transform_train, use_dataset_cache=True, num_samples= num_train_subset)

testset_kneePD =  SliceDataset(dataset = test_path_PD, path_to_dataset='', path_to_sensmaps=None, provide_senmaps=False, 
                challenge="multicoil", transform=data_transform_train, use_dataset_cache=True, num_samples= num_test_subset)
testset_kneePDFS =  SliceDataset(dataset = test_path_PDFS, path_to_dataset='', path_to_sensmaps=None, provide_senmaps=False, 
                challenge="multicoil", transform=data_transform_train, use_dataset_cache=True, num_samples= num_test_subset)
testset_brainFLAIR =  SliceDataset(dataset = test_path_FLAIR, path_to_dataset='', path_to_sensmaps=None, provide_senmaps=False, 
                challenge="multicoil", transform=data_transform_train, use_dataset_cache=True, num_samples= num_test_subset)
testset = torch.utils.data.ConcatDataset([testset_kneePD, testset_kneePDFS, testset_brainFLAIR])

# test dataloader
test_dataloader = torch.utils.data.DataLoader(dataset = testset, batch_size = 1, num_workers = 8, 
                    shuffle = False, generator = torch.Generator().manual_seed(1), pin_memory = True)


# model and maml
model = Unet(in_chans = 1,out_chans = 1,chans = 64, num_pool_layers = 4,drop_prob = 0.0)
#model = nn.DataParallel(model).to(device)
model = model.to(device)
#model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[0,1,2,3])

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
        #L2_vals_dc.append(loss.item())
        total_loss += loss
    evaluate_loss = total_loss / len(dataloader.dataset)
    evaluate_loss_history.append(evaluate_loss.item())
    return evaluate_loss


optimizer = optim.Adam(maml.parameters(), meta_lr)
#optimizer = torch.optim.RMSprop(model.parameters(),lr=0.0001,weight_decay=0.0)
lossfn = nn.MSELoss(reduction='sum')


test_loss_history = []

# outer loop
for out_iter in range(OUTER_EPOCH):    # time step
    print('Outer Loop Iteration:', out_iter+1)
    
    # split train and val set, val for calculating loss sum
    torch.manual_seed(out_iter)     # for reproducible
    train_set_kneePD, val_set_kneePD = torch.utils.data.random_split(trainset_kneePD, train_val_split)
    train_set_kneePDFS, val_set_kneePDFS = torch.utils.data.random_split(trainset_kneePDFS, train_val_split)
    train_set_brainFLAIR, val_set_brainFLAIR = torch.utils.data.random_split(trainset_brainFLAIR, train_val_split)
    
    train_set = torch.utils.data.ConcatDataset([train_set_kneePD, train_set_kneePDFS, train_set_brainFLAIR])
    val_set = torch.utils.data.ConcatDataset([val_set_kneePD, val_set_kneePDFS, val_set_brainFLAIR])

    # validation set dataloader
    update_dataloader = torch.utils.data.DataLoader(dataset = val_set, batch_size = 1, num_workers = 8, 
                    shuffle = False, generator = torch.Generator().manual_seed(1), pin_memory = True)

    # update real model
    model.train()
    total_update_loss = 0.0

    ###### inner loop ######
    # inner_iter could be any value: "using multiple gradient updates is a straightforward extension"
    # inner loop: adapt
    for inner_iter in range(INNER_EPOCH):      
        print('Inner Loop Iteration:', inner_iter+1)

        # sample k examples
        K_examples_dataloader = MAML_K_sampler(train_set, K)
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
        for _ in range(adapt_steps): # adaptation_steps
            K_examples_preds = learner(K_examples_inputs)
            adapt_loss = lossfn(K_examples_preds, K_examples_targets) / torch.sum(torch.abs(K_examples_targets)**2)
            print('Inner loop adapt training loss: ', adapt_loss.item())
            # 6. Compute  adapted  parameters  with  gradient  descent: θ′i = θ − α∇θLTi(fθ)
            learner.adapt(adapt_loss)
        
        # update loss
        update_loss = 0.0
        for iter, update_batch in enumerate(update_dataloader): 
            val_inputs = update_batch.image.unsqueeze(1).to(device)
            val_targets = update_batch.target.unsqueeze(1).to(device)
            val_output = learner(val_inputs)

            # NMSE
            val_loss = lossfn(val_output, val_targets) / torch.sum(torch.abs(val_targets)**2)
            # total val loss: validation_loss
            update_loss += val_loss

        print('Update training loss: ', update_loss.item()/(iter+1))
        total_update_loss += update_loss

    # update θ through validation tasks' loss
    total_update_loss = total_update_loss / (len(update_dataloader.dataset) * INNER_EPOCH)
    optimizer.zero_grad()
    update_loss.backward()
    optimizer.step()

    writer.add_scalar("Meta-update training NMSE", total_update_loss.item(), out_iter+1)
    
    # test
    if (out_iter+1) % 100 == 0:
        test_loss = 0.0
        test_loss = evaluate(model, test_dataloader, test_loss, test_loss_history)
        print('Testing loss: ', test_loss.item())
        writer.add_scalar("Testing NMSE", test_loss_history[int(out_iter/100)], out_iter+1)
    
    # save checkpoint per 1000 outer epoch
    if (out_iter+1) % 200 == 0:
        save_path_epoch = experiment_path + experiment_name + '_E' + str((out_iter+1)) + '.pth'
        torch.save((model.state_dict()), save_path_epoch)
        print('Model saved to', save_path_epoch)

### save model
save_path = experiment_path + experiment_name + '.pth'
torch.save((model.state_dict()), save_path)
print('Model saved to', save_path)