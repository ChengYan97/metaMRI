# %%
import yaml
import os
import numpy as np
import sys
import h5py
import matplotlib.pyplot as plt

### after you install bart 0.7.00 from https://mrirecon.github.io/bart/, import it as follows
sys.path.insert(0,'/tobit/bart-0.7.00/python/')
os.environ['TOOLBOX_PATH'] = "/tobit/bart-0.7.00/"
import bart

# %%

# %%
def apply_center_mask(kspace,center_fraction):
    num_cols = kspace.shape[-1]

    #center_fraction = 0.08
    acceleration = 4

    # create the mask
    num_low_freqs = int(round(num_cols * center_fraction))
    #print(num_low_freqs)
    num_high_freqs = round(num_cols / acceleration) - num_low_freqs
    pad = (num_cols - num_low_freqs + 1) // 2

    all_freqs_indices = np.arange(num_cols)
    all_high_freqs_indices = np.hstack((all_freqs_indices[: pad],all_freqs_indices[pad + num_low_freqs :]))

    chosen_high_freqs_indices = np.random.choice(all_high_freqs_indices, size=num_high_freqs, replace=False, p=None)

    mask = np.zeros(num_cols)
    
    mask[pad : pad + num_low_freqs] = 1.0
    #mask[chosen_high_freqs_indices] = 1.0
    #print(mask)
    
    mask_shape = [1 for _ in kspace.shape]
    mask_shape[-1] = num_cols
    mask = mask.reshape(*mask_shape)

    masked_kspace = kspace*mask + 0.0
    return masked_kspace, num_low_freqs

def compute_sensitivity_maps(save_path,filenames):
    counter = 0
    for file in filenames:
        
        ## Load kspace
        with h5py.File(file["path"].replace("hdd1", "ssd1"), "r") as hf:
            kspace_slice = hf["kspace"][file["slice"]]
        
        center_fraction = 0.08

        counter+=1
        print(counter)

        ## Apply mask only on the 0.08 center fraction
        masked_kspace, num_low_freqs = apply_center_mask(kspace_slice,center_fraction)

        ## Compute sens map
        sens_maps = bart.bart(1, f'ecalib -d0 -m1', np.array([np.moveaxis(masked_kspace,0,2)]))
        sens_maps = np.moveaxis(sens_maps[0],2,0)   

        ## Save sens map
        sensmap_fname = file["filename"] + '_smaps_slice' + str(file["slice"]) + '.h5'
        with h5py.File(save_path + sensmap_fname, "w") as hf:
            hf.create_dataset('sens_maps', data=sens_maps)  

# %%
## Saves sens maps as a single .h5 file per slice

with open('/tobit/metaMRI/data_dict/TTT_paper/TTT_brain_train_300.yaml', 'r') as stream:
    filenames = yaml.safe_load(stream)

print(filenames[0:3])

save_path = '/tobit/metaMRI/data_dict/TTT_paper/sensmap_brain_train/'

compute_sensitivity_maps(save_path,filenames)

# %%
## Saves sens maps as a single .h5 file per slice

with open('/tobit/metaMRI/data_dict/TTT_paper/TTT_brain_val.yaml', 'r') as stream:
    filenames = yaml.safe_load(stream)

print(filenames[0:3])

save_path = '/tobit/metaMRI/data_dict/TTT_paper/sensmap_brain_val/'

compute_sensitivity_maps(save_path,filenames)# %%
## Saves sens maps as a single .h5 file per slice

with open('/tobit/metaMRI/data_dict/TTT_paper/TTT_knee_train_300.yaml', 'r') as stream:
    filenames = yaml.safe_load(stream)

print(filenames[0:3])

save_path = '/tobit/metaMRI/data_dict/TTT_paper/sensmap_knee_train/'

compute_sensitivity_maps(save_path,filenames)# %%
## Saves sens maps as a single .h5 file per slice

with open('/tobit/metaMRI/data_dict/TTT_paper/TTT_knee_val.yaml', 'r') as stream:
    filenames = yaml.safe_load(stream)

print(filenames[0:3])

save_path = '/tobit/metaMRI/data_dict/TTT_paper/sensmap_knee_val/'

compute_sensitivity_maps(save_path,filenames)

# %%
#print(sens_maps.shape)

# %%
#fig = plt.figure(figsize=(10,10))
#ax = fig.add_subplot(111)
#ax.imshow(np.abs(sens_maps[0]),'gray')
#ax.axis('off')

# %%



