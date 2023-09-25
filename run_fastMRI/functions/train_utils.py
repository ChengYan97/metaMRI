
import torch
import numpy as np
import random
import os
import glob
import logging
from torch.serialization import default_restore_location
from tensorboard.backend.event_processing import event_accumulator
import time
import matplotlib.pyplot as plt

def setup_experiment(hp_exp):
    '''
    - Handle seeding
    - Create directories
    - Look for checkpoints to load from
    '''
    torch.backends.cudnn.deterministic = True

    # Disabling the benchmarking feature with torch.backends.cudnn.benchmark = False causes cuDNN to deterministically select an algorithm, 
    # possibly at the cost of reduced performance.
    # However, if you do not need reproducibility across multiple executions of your application, 
    # then performance might improve if the benchmarking feature is enabled with torch.backends.cudnn.benchmark = True.
    torch.backends.cudnn.benchmark = False
    
    torch.manual_seed(hp_exp['seed'])
    torch.cuda.manual_seed(hp_exp['seed'])
    np.random.seed(hp_exp['seed'])
    random.seed(hp_exp['seed'])

    hp_exp['log_path'] = '../'+ hp_exp['exp_name'] + '/log_files/'
    os.makedirs(hp_exp['log_path'] + 'checkpoints/', exist_ok=True)
    
    hp_exp['log_file'] = os.path.join(hp_exp['log_path'], "train.log")

    # Look for checkpoints to load from
    available_models = glob.glob(hp_exp['log_path'] + 'checkpoints/*.pt')
    if available_models and hp_exp['resume_from_which_checkpoint']=='last':
        hp_exp['restore_file'] = hp_exp['log_path'] + 'checkpoints/checkpoint_last.pt'
    elif available_models and hp_exp['resume_from_which_checkpoint']=='best':
        hp_exp['restore_file'] = hp_exp['log_path'] + 'checkpoints/checkpoint_best.pt'
    else:
        hp_exp['restore_file'] = None


    # Set attributes of the function save_checkpoint. They will be used to track the validation score and trigger saving a checkpoint
    mode_lookup = {
        'SSIM' : 'max',
        'PSNR' : 'max',
        'L1' : 'min',
        'L2' : 'min',
        'MSE' : 'min',
        'L2_kspace' : 'min',
        'L1_kspace' : 'min',
    }
    
    save_checkpoint.best_epoch = -1
    save_checkpoint.last_epoch = 0
    save_checkpoint.start_epoch = 0
    save_checkpoint.global_step = 0
    save_checkpoint.current_lr = hp_exp['lr']
    save_checkpoint.break_counter = 0
    #save_checkpoint.break_annealing_counter = 0
    save_checkpoint.best_val_current_lr_interval = float("inf") if  mode_lookup[hp_exp['decay_metric']] == "min" else float("-inf")
    save_checkpoint.lr_interval_counter = 0
    #if hp_exp['lr_annealing']:
    #    save_checkpoint.start_decay = False
    #    save_checkpoint.mode = mode_lookup[hp_exp['anneal_metric']]
    #    save_checkpoint.best_score =  float("inf") if save_checkpoint.mode == "min" else float("-inf")
    #else:
    #save_checkpoint.start_decay = True
    save_checkpoint.mode = mode_lookup[hp_exp['decay_metric']]
    save_checkpoint.best_score =  float("inf") if save_checkpoint.mode == "min" else float("-inf")

    return hp_exp

def init_logging(hp_exp):
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    handlers = [logging.StreamHandler()]

    mode = "a" if hp_exp['restore_file'] else "w"
    handlers.append(logging.FileHandler(hp_exp['log_file'], mode=mode))
    logging.basicConfig(handlers=handlers, format="[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
    if hp_exp['mode'] == 'train':
        logging.info("Arguments: {}".format(hp_exp))
        


def save_checkpoint(hp_exp, epoch, model, optimizer=None, scheduler=None, score=None):
    ''''
    This function is used to save a range of parameters related to the training progress. 
    Saving those parameters allows to interrupt and then pick up training later at any point.
    At the beginning of every experiment the parameters are initialized in setup_experiment()
    Parameters:
    - best_score: Holds the best validation score so far
    - best_epoch: Holds the epoch in which the best validation score was achieved
    - last_epoch: Holds the current epoch.
    - break_counter: Count the number of epochs with minimal lr
    - best_val_current_lr_interval: Holds the best val performance for the current lr-inerval
    - lr_interval_counter: Counts for how many lr intervals there was no improvement
    - 
    '''
    save_checkpoint.last_epoch = epoch
    best_score = save_checkpoint.best_score
    
    if (score < best_score and save_checkpoint.mode == "min") or (score > best_score and save_checkpoint.mode == "max"):
        save_checkpoint.best_epoch = epoch
        save_checkpoint.best_score = score

    model = [model] if model is not None and not isinstance(model, list) else model
    optimizer = [optimizer] if optimizer is not None and not isinstance(optimizer, list) else optimizer
    scheduler = [scheduler] if scheduler is not None and not isinstance(scheduler, list) else scheduler
    state_dict = {
        "last_step": save_checkpoint.global_step, #set
        "last_score": score, #set
        "break_counter": save_checkpoint.break_counter,
        "best_val_current_lr_interval": save_checkpoint.best_val_current_lr_interval,
        "lr_interval_counter": save_checkpoint.lr_interval_counter,
        "last_epoch": save_checkpoint.last_epoch, #set
        "best_epoch": save_checkpoint.best_epoch, #set
        #"start_decay": save_checkpoint.start_decay,
        "current_lr":save_checkpoint.current_lr, #set
        #"break_annealing_counter": save_checkpoint.break_annealing_counter, #set
        "mode": save_checkpoint.mode,
        "best_score": getattr(save_checkpoint, "best_score", None), #set
        "model": [m.state_dict() for m in model] if model is not None else None,
        "optimizer": [o.state_dict() for o in optimizer] if optimizer is not None else None,
        "scheduler": [s.state_dict() for s in scheduler] if scheduler is not None else None,
        "args": hp_exp,
    }
    torch.save(state_dict, os.path.join(hp_exp['log_path'] + 'checkpoints/', "checkpoint_last.pt"))

    if hp_exp['epoch_checkpoints']:
        if epoch in hp_exp['epoch_checkpoints']:
            torch.save(state_dict, os.path.join(hp_exp['log_path'] + 'checkpoints/', "checkpoint{}.pt".format(epoch)))
    if (score < best_score and save_checkpoint.mode == "min") or (score > best_score and save_checkpoint.mode == "max"):
        torch.save(state_dict, os.path.join(hp_exp['log_path'] + 'checkpoints/', "checkpoint_best.pt"))

            
            
def load_checkpoint(hp_exp, model=None, optimizer=None, scheduler=None):
    
    print('restoring model..')
    state_dict = torch.load(hp_exp['restore_file'], map_location=lambda s, l: default_restore_location(s, "cpu"))

    save_checkpoint.last_epoch = state_dict["last_epoch"]
    save_checkpoint.start_epoch = state_dict["last_epoch"]+1
    save_checkpoint.global_step = state_dict["last_step"]
    save_checkpoint.best_score = state_dict["best_score"]
    save_checkpoint.best_epoch = state_dict["best_epoch"]
    save_checkpoint.break_counter = state_dict["break_counter"]
    save_checkpoint.best_val_current_lr_interval = state_dict["best_val_current_lr_interval"]
    save_checkpoint.lr_interval_counter = state_dict["lr_interval_counter"]
    #save_checkpoint.start_decay = state_dict["start_decay"]
    save_checkpoint.current_lr = state_dict["current_lr"]
    #save_checkpoint.break_annealing_counter = state_dict["break_annealing_counter"]
    save_checkpoint.mode = state_dict["mode"]

    #print(save_checkpoint.break_counter, save_checkpoint.current_lr, hp_exp['lr_min'])
    

    model = [model] if model is not None and not isinstance(model, list) else model
    optimizer = [optimizer] if optimizer is not None and not isinstance(optimizer, list) else optimizer
    scheduler = [scheduler] if scheduler is not None and not isinstance(scheduler, list) else scheduler
    if model is not None and state_dict.get("model", None) is not None:
        for m, state in zip(model, state_dict["model"]):
            m.load_state_dict(state)
    if optimizer is not None and state_dict.get("optimizer", None) is not None:
        for o, state in zip(optimizer, state_dict["optimizer"]):
            o.load_state_dict(state)
    if scheduler is not None and state_dict.get("scheduler", None) is not None:
        for s, state in zip(scheduler, state_dict["scheduler"]):
            #milestones = s.milestones
            #state['milestones'] = milestones
            s.load_state_dict(state)
            #s.milestones = milestones

    logging.info("Loaded checkpoint {} from best_epoch {} last_epoch {}".format(hp_exp['restore_file'], save_checkpoint.best_epoch, save_checkpoint.last_epoch))

def get_event_details(exp, train_meters, val_metric_dict, hp_exp):
    print(exp)
    exception_list = []
    if not any(exception in exp for exception in exception_list):
        log_path = '../' + exp + '/log_files/'
        if os.path.exists(log_path): 
            tf_list = glob.glob(log_path+'events*')

            #if tf_list:
            #    for tf in tf_list:
            #        print(tf)
            #        ind1 = str(version).find('version_')
            #        ind2 = str(tf).find('events')
            #        filename = str(version)[ind1:] + '_' + str(tf)[ind2:]
            #        print(filename)
            #        shutil.move(tf, out_path/filename)  # For Python 3.8+.
                        
                        
            
            events = sorted(tf_list, key=os.path.getmtime)
            print(events)
            
            if events:
                ts = int(exp[exp.find('_t')+2:exp.find('_l')])
                # Get batch size
                bs = hp_exp['batch_size']
                
                epoch = np.zeros((1,3))
                lr = np.zeros((1,3))

                valid_metrics = {}
                train_metrics = {}
                for val_loss_name in val_metric_dict.keys():
                    valid_metrics['val_'+val_loss_name] = np.zeros((1,3))

                for tr_loss_name in train_meters.keys():
                    train_metrics[tr_loss_name] = np.zeros((1,3))

                #ssim_valid = np.zeros((1,3))
                #psnr_valid = np.zeros((1,3))
                #L1_valid = np.zeros((1,3))
                #L2_valid = np.zeros((1,3))
                #loss_train = np.zeros((1,3))
                

                for event in events:
                    start = time.process_time()
                    ea = event_accumulator.EventAccumulator(event,
                        size_guidance={event_accumulator.SCALARS: 0,event_accumulator.IMAGES: 1,})
                    ea.Reload()
                    
                    # Get all other data as numpy arrays of size num_epochs x 3, with epochs in first column, values in second column and time stamp in third column
                    w_times, step_nums, vals = zip(*ea.Scalars('epoch'))
                    epoch_tmp = np.vstack((np.array(step_nums), np.array(vals),np.array(w_times))).T
                    epoch = np.vstack((epoch,epoch_tmp))

                    w_times, step_nums, vals = zip(*ea.Scalars('lr'))
                    lr_tmp = np.vstack((np.array(step_nums), np.array(vals),np.array(w_times))).T
                    lr = np.vstack((lr,lr_tmp))

                    for val_loss_name in valid_metrics.keys():
                        w_times, step_nums, vals = zip(*ea.Scalars(val_loss_name))
                        tmp = np.vstack((np.array(step_nums), np.array(vals),np.array(w_times))).T
                        valid_metrics[val_loss_name] = np.vstack((valid_metrics[val_loss_name],tmp))

                    for tr_loss_name in train_metrics.keys():
                        w_times, step_nums, vals = zip(*ea.Scalars(tr_loss_name))
                        tmp = np.vstack((np.array(step_nums), np.array(vals),np.array(w_times))).T
                        train_metrics[tr_loss_name] = np.vstack((train_metrics[tr_loss_name],tmp))                   

                    end = time.process_time() - start
                    print(np.round(end/60,3))

                epoch = epoch[1:,:]
                lr = lr[1:,:]
                for val_loss_name in valid_metrics.keys():
                    valid_metrics[val_loss_name] = valid_metrics[val_loss_name][1:,:]

                for tr_loss_name in train_metrics.keys():
                    train_metrics[tr_loss_name] = train_metrics[tr_loss_name][1:,:]
                

                # Compute stats
                mode_lookup = {
                        'SSIM' : 'max',
                        'PSNR' : 'max',
                        'L1' : 'min',
                        'L2' : 'min',
                        'MSE' : 'min',
                        'L2_kspace' : 'min',
                        'L1_kspace' : 'min',
                    }
                mode = mode_lookup[hp_exp['decay_metric']]

                num_epochs = epoch[-1,1]+1
                if mode == 'max':
                    best_epoch = np.where(valid_metrics['val_'+hp_exp['decay_metric']][:,1]==np.max(valid_metrics['val_'+hp_exp['decay_metric']][:,1]))[0][0]+1
                else:
                    best_epoch = np.where(valid_metrics['val_'+hp_exp['decay_metric']][:,1]==np.min(valid_metrics['val_'+hp_exp['decay_metric']][:,1]))[0][0]+1

                steps_per_epoch = np.ceil(ts/bs)
                gpu_hours = np.min(epoch[1:,2]-epoch[0:-1,2])*num_epochs/(60*60)
                print(gpu_hours)     

                #if hp_exp['lr_annealing']:
                #    base_lr = np.max(lr[:,1]) * hp_exp['lr_inital_decay_factor'] 
                #else:
                base_lr = hp_exp['lr']

                # Save to dict
                stats_dict = {}
                stats_dict['exp_name'] = exp
                stats_dict['batch_size'] = bs
                stats_dict['train_size'] = ts
                stats_dict['steps_per_epoch'] = steps_per_epoch
                stats_dict['num_epochs'] = num_epochs
                stats_dict['best_epoch'] = best_epoch
                stats_dict['lr'] = lr
                stats_dict['epoch'] = epoch
                stats_dict['gpu_hours'] = gpu_hours
                stats_dict['base_lr'] = base_lr
                stats_dict['hp_exp'] = hp_exp

                for val_loss_name in valid_metrics.keys():
                    stats_dict[val_loss_name] = valid_metrics[val_loss_name]

                for tr_loss_name in train_metrics.keys():
                    stats_dict[tr_loss_name] = train_metrics[tr_loss_name]

                np.save(log_path+"/tb_events_dict.npy", stats_dict)

                # If the training metric is also computed at validation, save a figure containig the training and the validation loss.
                # Otherwise plot separate figures.
                for train_loss_name in hp_exp['loss_functions']:
                    if train_loss_name in [key for key in val_metric_dict.keys()]:
                        train_loss_over_ep = train_metrics['train_' + train_loss_name][:,1]
                        if train_loss_name == 'SSIM':
                            train_loss_over_ep = 1 - train_loss_over_ep
                        val_loss_over_ep = valid_metrics['val_' + train_loss_name][:,1]
                        epochs = epoch[:,1]
                        fig = plt.figure(figsize=(6,4))
                        ax = fig.add_subplot(1,1,1)
                        ax.plot(epochs,train_loss_over_ep,label="train")
                        ax.plot(epochs,val_loss_over_ep,label="val")
                        ax.set_ylabel(train_loss_name)
                        ax.set_xlabel('Epochs')
                        ax.legend()

                        ax2 = ax.twinx()
                        ax2.plot(epochs,lr[:,1], label='lr', color='r', alpha=0.3)
                        ax2.set_ylabel('learning rate')
                        ax2.set_yscale('log')
                        #ax2.legend()

                        fig.tight_layout()

                        if not os.path.isdir(hp_exp['log_path'] + 'loss_curve_plots/'):
                            os.mkdir(hp_exp['log_path'] + 'loss_curve_plots/')
                        fig.savefig(hp_exp['log_path'] + 'loss_curve_plots/train_val_lr_over_epochs.png')



            else:
                print('This experiment is on a different server.')


