import random
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
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

experiment_name = 'E1.1_few-tuning_kneePD_50epoch'

# tensorboard dir
experiment_path = '/cheng/metaMRI/metaMRI/save/' + experiment_name + '/'
writer = SummaryWriter(experiment_path)

TRAINING_EPOCH = 50
num_sample_train = 10
num_sample_val = 200
path_PDFS_train = '/cheng/metaMRI/metaMRI/data_dict/knee_multicoil_train_CORPD.yaml'
path_PDFS_val = '/cheng/metaMRI/metaMRI/data_dict/knee_multicoil_val_CORPD.yaml'
checkpoint_path = '/cheng/metaMRI/metaMRI/save/E1.1/E1.1_train_1dataset500_70epoch/E1.1_train_1dataset500_70epoch.pth'

# mask function and data transform
mask_function = create_mask_for_mask_type(mask_type_str = 'random', self_sup = False, 
                    center_fraction = 0.08, acceleration = 4.0, acceleration_total = 3.0)

data_transform_train = UnetDataTransform('multicoil', mask_func = mask_function, use_seed = False)
data_transform_test = UnetDataTransform('multicoil', mask_func = mask_function, use_seed=True)

# dataset: num_sample_subset x 3
trainset = SliceDataset(dataset = path_PDFS_train, path_to_dataset='', path_to_sensmaps=None, provide_senmaps=False, 
                challenge="multicoil", transform=data_transform_train, use_dataset_cache=True, num_samples= num_sample_train)

testset = SliceDataset(dataset = path_PDFS_train, path_to_dataset='', path_to_sensmaps=None, provide_senmaps=False, 
                    challenge="multicoil", transform=data_transform_train, use_dataset_cache=True, num_samples= num_sample_val)

# dataloader: batch size 1 
train_dataloader = torch.utils.data.DataLoader(dataset = trainset, batch_size = 1, num_workers = 8, 
                    shuffle = True, generator = torch.Generator().manual_seed(1), pin_memory = True)
test_dataloader = torch.utils.data.DataLoader(dataset = testset, batch_size = 1, num_workers = 8, 
                    shuffle = False, generator = torch.Generator().manual_seed(1), pin_memory = True)


def train(model, dataloader, optimizer, train_loss_history): 
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
    train_loss_history.append(avg_train_loss)
    print('Training Loss', avg_train_loss) 


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

model = Unet(in_chans=1, out_chans=1, chans=64, num_pool_layers=4, drop_prob=0.0)
model.load_state_dict(torch.load(checkpoint_path))
model = model.to(device)


##########################
# optimizer = torch.optim.Adam(model.parameters(),lr=0.001, betas=(0.9, 0.999), 
#                              eps=1e-08, weight_decay=0.0, amsgrad=False)
optimizer = torch.optim.Adam(model.parameters(),lr=0.0001)
# optimizer = torch.optim.RMSprop(model.parameters(),lr=0.0001,weight_decay=0.0)
#scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-8)

train_loss_history = []
val_loss_history = []
for iteration in range(TRAINING_EPOCH):
    print('Iteration:', iteration+1)
    # training
    train(model, train_dataloader, optimizer, train_loss_history)
    writer.add_scalar("Training NMSE", train_loss_history[iteration], iteration)
    # val
    evaluate(model, test_dataloader, val_loss_history)
    writer.add_scalar("Validation NMSE", val_loss_history[iteration], iteration)
    # 
    #scheduler.step(val_loss_history[-1])

### save model
save_path = '/cheng/metaMRI/metaMRI/save/'+ experiment_name + '/' + experiment_name + '.pth'
torch.save((model.state_dict()), save_path)
print('Model saved to', save_path)
