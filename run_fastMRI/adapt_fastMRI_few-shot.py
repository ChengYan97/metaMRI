#%%
import random
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import learn2learn as l2l
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
# The corase reconstruction is the rss of the zerofilled multi-coil kspaces
# after inverse FT.
from functions.data.transforms import UnetDataTransform_norm
from functions.data.mri_dataset import SliceDataset
from functions.models.unet import Unet
from functions.data.subsample import create_mask_for_mask_type
from functions.training.losses import SSIMLoss
from functions.helper import average_early_stopping_epoch, evaluate_loss_dataloader, adapt

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

####################################################################################
SEED = 1            # 1,2,3,4,5 repeat # for Q2 using seed 1,3,4,5,6
INIT = 'maml'       # 'standard', 'maml'
TARGET = 'P8_train'       # 'Q1', 'Q2', 'Q3'
adapt_shot = 200
TRAINING_EPOCH = 10
LR = 1e-3 
cosine_annealing = True
early_stopping = False

if cosine_annealing: 
    experiment_name = "E_sup_" + INIT + "_" + str(adapt_shot) + "few-adapt_lrCA" + str(LR) + "_" + TARGET +'_seed' + str(SEED)
else: 
    experiment_name = "E_sup_" + INIT + "_" + str(adapt_shot) + "few-adapt_lr" + str(LR) + "_" + TARGET +'_seed' + str(SEED)
print('Experiment: ', experiment_name)
# tensorboard dir
experiment_path = '/cheng/metaMRI/metaMRI/save/' + experiment_name + '/'
writer = SummaryWriter(experiment_path)

####################################################################################
# seed
random.seed(SEED)
np.random.seed(SEED)
torch.cuda.manual_seed(SEED)
torch.manual_seed(SEED)

# different trained weight
if INIT == 'standard_5batchsize':
    checkpoint_path = "/cheng/metaMRI/metaMRI/save/E6*/E6.6_standard(NMSE-lrAnneal)_T8x200_100epoch_E85_best.pth"
elif INIT == 'standard':
    checkpoint_path = "/cheng/metaMRI/metaMRI/thesis/metaMRI/setup_8knee/checkpoints/E6.6+_standard(NMSE-lrAnneal)_T8x200_120epoch/E6.6+_standard(NMSE-lrAnneal)_T8x200_120epoch_E64_best.pth"
elif INIT == 'maml':
    checkpoint_path = "/cheng/metaMRI/metaMRI/thesis/metaMRI/setup_8knee/checkpoints/E6.4_maml(NMSE-lr-in1e-3-out1e-4)_T8x200_200epoch_E200_best.pth"
elif INIT == 'maml_AS5':
    checkpoint_path = "/cheng/metaMRI/metaMRI/save/E_MAML(NMSE-out-3-in-4)_T8x200knee_200epoch/E_MAML(NMSE-out-3-in-4)_T8x200knee_200epoch_E196.pth"
elif INIT == 'maml_AS9':
    checkpoint_path = "/cheng/metaMRI/metaMRI/save/E_MAML(NMSE_out-3-in-4_AS-9)_T8x200knee_300epoch/E_MAML(NMSE_out-3-in-4_AS-9)_T8x200knee_300epoch_E191_best.pth"
elif INIT == 'standardE10.2':
    checkpoint_path = "/cheng/metaMRI/metaMRI/save/E10.2_standard(NMSE-lr1e-3CA4)_T8x200_120epoch/E10.2_standard(NMSE-lr1e-3CA4)_T8x200_120epoch_E87_best.pth"
elif INIT == 'standardE10.3':
    checkpoint_path = "/cheng/metaMRI/metaMRI/save/Conclusion 1.2 (E10.3)/E10.3_standard(NMSE-lr1e-3CA4)_T8x200_200epoch/E10.3_standard(NMSE-lr1e-3CA4)_T8x200_200epoch_E64_best.pth"
elif INIT == 'mamlE10.2':
    checkpoint_path = "/cheng/metaMRI/metaMRI/save/E10.2_maml(NMSE-lre-3)_T8x200_250epoch/E10.2_maml(NMSE-lre-3)_T8x200_250epoch_E238_best.pth"
elif INIT == 'mamlE10.3':
    checkpoint_path = "/cheng/metaMRI/metaMRI/save/E10.3_maml(NMSE-lre-3)_T8x200_250epoch/E10.3_maml(NMSE-lre-3)_T8x200_250epoch_E98_best.pth"
elif INIT == 'mamlE10.4':
    checkpoint_path = "/cheng/metaMRI/metaMRI/save/E10.4_maml(NMSE-lre-3)_T8x50_250epoch/E10.4_maml(NMSE-lre-3)_T8x50_250epoch_E247_best.pth"
elif INIT == 'standardE10.4':
    checkpoint_path = "/cheng/metaMRI/metaMRI/save/E10.4_standard(NMSE-lr1e-3CA4)_T8x50_200epoch/E10.4_standard(NMSE-lr1e-3CA4)_T8x50_200epoch_E93_best.pth"
elif INIT == 'maml_12mix':
    checkpoint_path = '/cheng/metaMRI/metaMRI/save/E_MAML(NMSE-out-3-in-4)_T12x200mix_300epoch/E_MAML(NMSE-out-3-in-4)_T12x200mix_300epoch_E300.pth'
elif INIT == 'standard_12mix':
    checkpoint_path = '/cheng/metaMRI/metaMRI/save/E_standard(NMSE-lr1e-3CA4)_T12x200mix_120epoch/E_standard(NMSE-lr1e-3CA4)_T12x200mix_120epoch_E69_best.pth'
elif INIT == 'fomaml_8knee':
    checkpoint_path = "/cheng/metaMRI/metaMRI/save/E_FOMAML(NMSE_outCA-3-4-in-4)_T8x200knee_300epoch/E_FOMAML(NMSE_outCA-3-4-in-4)_T8x200knee_300epoch_E191_best.pth"
elif INIT == 'maml_in-3_8knee':
    checkpoint_path = "/cheng/metaMRI/metaMRI/save/E_MAML(NMSE_outCA-3-4-in-3_AS-9)_T8x200knee_300epoch/E_MAML(NMSE_outCA-3-4-in-3_AS-9)_T8x200knee_300epoch_E196_best.pth"
else: 
    print('Choose the initialization weight. ')

# target domain
if TARGET == 'Q1': 
    path_adapt = '/cheng/metaMRI/metaMRI/data_dict/Task_8knee/Q/brain_train_T1POST_TrioTim_5-8.yaml'
    path_test = '/cheng/metaMRI/metaMRI/data_dict/Task_8knee/Q/brain_test_T1POST_TrioTim_5-8.yaml'
elif TARGET == 'Q2': 
    path_adapt = '/cheng/metaMRI/metaMRI/data_dict/Task_8knee/Q/brain_train_FLAIR_Skyra_5-8.yaml'
    path_test = '/cheng/metaMRI/metaMRI/data_dict/Task_8knee/Q/brain_test_FLAIR_Skyra_5-8.yaml'
elif TARGET == 'Q3': 
    path_adapt = '/cheng/metaMRI/metaMRI/data_dict/Task_8knee/Q/brain_train_T2_Aera_5-8.yaml'
    path_test = '/cheng/metaMRI/metaMRI/data_dict/Task_8knee/Q/brain_test_T2_Aera_5-8.yaml'
elif TARGET == 'Q4': 
    path_adapt = '/cheng/metaMRI/metaMRI/data_dict/E10.2/Q/brain_train_T1_Aera_5-8.yaml'
    path_test = '/cheng/metaMRI/metaMRI/data_dict/E10.2/Q/brain_test_T1_Aera_5-8.yaml'
elif TARGET == 'Q5': 
    path_adapt = '/cheng/metaMRI/metaMRI/data_dict/E10.2/Q/brain_train_T1POST_Avanto_5-8.yaml'
    path_test = '/cheng/metaMRI/metaMRI/data_dict/E10.2/Q/brain_test_T1POST_Avanto_5-8.yaml'
elif TARGET == 'P1_train': 
    path_adapt = '/cheng/metaMRI/metaMRI/data_dict/Task_8knee/P/knee_train_PD_Aera_2-9.yaml'
    path_test = '/cheng/metaMRI/metaMRI/data_dict/Task_8knee/P/knee_test_PD_Aera_2-9.yaml'
elif TARGET == 'P2_train': 
    path_adapt = '/cheng/metaMRI/metaMRI/data_dict/Task_8knee/P/knee_train_PD_Aera_15-22.yaml'
    path_test = '/cheng/metaMRI/metaMRI/data_dict/Task_8knee/P/knee_test_PD_Aera_15-22.yaml'
elif TARGET == 'P3_train': 
    path_adapt = '/cheng/metaMRI/metaMRI/data_dict/Task_8knee/P/knee_train_PD_Biograph_15-22.yaml'
    path_test = '/cheng/metaMRI/metaMRI/data_dict/Task_8knee/P/knee_test_PD_Biograph_15-22.yaml'
elif TARGET == 'P4_train': 
    path_adapt = '/cheng/metaMRI/metaMRI/data_dict/Task_8knee/P/knee_train_PD_Skyra_15-22.yaml'
    path_test = '/cheng/metaMRI/metaMRI/data_dict/Task_8knee/P/knee_test_PD_Skyra_15-22.yaml'
elif TARGET == 'P5_train': 
    path_adapt = '/cheng/metaMRI/metaMRI/data_dict/Task_8knee/P/knee_train_PDFS_Aera_2-9.yaml'
    path_test = '/cheng/metaMRI/metaMRI/data_dict/Task_8knee/P/knee_test_PDFS_Aera_2-9.yaml'
elif TARGET == 'P6_train': 
    path_adapt = '/cheng/metaMRI/metaMRI/data_dict/Task_8knee/P/knee_train_PDFS_Aera_15-22.yaml'
    path_test = '/cheng/metaMRI/metaMRI/data_dict/Task_8knee/P/knee_test_PDFS_Aera_15-22.yaml'
elif TARGET == 'P7_train': 
    path_adapt = '/cheng/metaMRI/metaMRI/data_dict/Task_8knee/P/knee_train_PDFS_Biograph_15-22.yaml'
    path_test = '/cheng/metaMRI/metaMRI/data_dict/Task_8knee/P/knee_test_PDFS_Biograph_15-22.yaml'
elif TARGET == 'P8_train': 
    path_adapt = '/cheng/metaMRI/metaMRI/data_dict/Task_8knee/P/knee_train_PDFS_Skyra_15-22.yaml'
    path_test = '/cheng/metaMRI/metaMRI/data_dict/Task_8knee/P/knee_test_PDFS_Skyra_15-22.yaml'
elif TARGET == '12P1_train': 
    path_adapt = '/cheng/metaMRI/metaMRI/data_dict/final/P1/knee_train_PD_Aera_5-9.yaml'
    path_test = '/cheng/metaMRI/metaMRI/data_dict/final/P1/knee_test_PD_Aera_5-9.yaml'
elif TARGET == '12P9_train': 
    path_adapt = '/cheng/metaMRI/metaMRI/data_dict/final/P9/brain_train_AXT1PRE_Skyra_1-5.yaml'
    path_test = '/cheng/metaMRI/metaMRI/data_dict/final/P9/brain_test_AXT1PRE_Skyra_1-5.yaml'


print(path_test)
#############################################################################################
### data
#############################################################################################
# mask function and data transform
mask_function = create_mask_for_mask_type(mask_type_str = 'random', self_sup = False, 
                    center_fraction = 0.08, acceleration = 4.0, acceleration_total = 3.0)


data_transform = UnetDataTransform_norm('multicoil', mask_func = mask_function, use_seed=True, mode='adapt')

sourse_set = SliceDataset(dataset = path_adapt, path_to_dataset='', path_to_sensmaps=None, provide_senmaps=False, 
                        challenge="multicoil", transform=data_transform, use_dataset_cache=True)
indices = random.sample(range(len(sourse_set)), adapt_shot)
adapt_dataset = torch.utils.data.Subset(sourse_set, indices)
adapt_dataloader = torch.utils.data.DataLoader(adapt_dataset, batch_size = 1, num_workers = 8, 
                            shuffle = True, generator = torch.Generator().manual_seed(SEED), pin_memory = True)
#%%

print("Adapt data number: ", len(adapt_dataloader))

test_set = SliceDataset(dataset = path_test, path_to_dataset='', path_to_sensmaps=None, provide_senmaps=False, 
                      challenge="multicoil", transform=data_transform, use_dataset_cache=True)
print("Test data number: ", len(test_set))

# dataloader: batch size 1 
test_dataloader = torch.utils.data.DataLoader(dataset = test_set, batch_size = 1, num_workers = 8, 
                    shuffle = True, generator = torch.Generator().manual_seed(SEED))

#############################################################################################
### model
#############################################################################################
model = Unet(in_chans=1, out_chans=1, chans=32, num_pool_layers=4, drop_prob=0.0)
model.load_state_dict(torch.load(checkpoint_path))
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(),lr=LR)
if cosine_annealing: 
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, TRAINING_EPOCH/1, eta_min=0.0001, last_epoch=-1)

#############################################################################################
### few-shots adaptation
#############################################################################################

if early_stopping:
    suggest_epoch = average_early_stopping_epoch(model, checkpoint_path, adapt_dataset, training_epoch=30, split_ratio=[7,3], repeat_times=5)
    print('The early-stopping epoch: ', suggest_epoch)
    # re-initialize the model
    model.load_state_dict(torch.load(checkpoint_path))
else: 
    pass


test_loss_l1, test_loss_NMSE, test_loss_PSNR, test_loss_SSIM = evaluate_loss_dataloader(model, test_dataloader)
writer.add_scalar("Testing NMAE", test_loss_l1, 0)
writer.add_scalar("Testing NMSE", test_loss_NMSE, 0)
writer.add_scalar("Testing PSNR", test_loss_PSNR, 0)
writer.add_scalar("Testing SSIM", test_loss_SSIM, 0)

for iteration in range(TRAINING_EPOCH):
    print('Iteration:', iteration+1)
    # training
    training_loss = adapt(model, adapt_dataloader, optimizer)
    writer.add_scalar("Adaptation training NMSE", training_loss, iteration+1)
    if cosine_annealing: 
        scheduler.step()
    # val
    test_loss_l1, test_loss_NMSE, test_loss_PSNR, test_loss_SSIM = evaluate_loss_dataloader(model, test_dataloader)
    writer.add_scalar("Testing NMAE", test_loss_l1, iteration+1)
    writer.add_scalar("Testing NMSE", test_loss_NMSE, iteration+1)
    writer.add_scalar("Testing PSNR", test_loss_PSNR, iteration+1)
    writer.add_scalar("Testing SSIM", test_loss_SSIM, iteration+1)

