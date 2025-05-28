import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms

from sklearn.model_selection import train_test_split

import os, sys
import numpy as np
import random

# additional
from collections import Counter
from torch.utils.data import WeightedRandomSampler
import math

class Load_Dataset(Dataset):
    def __init__(self, dataset, dataset_configs):
        super().__init__()
        self.num_channels = dataset_configs.input_channels

        # Load samples
        x_data = dataset["samples"]

        # Load labels
        y_data = dataset.get("labels")
        if y_data is not None and isinstance(y_data, np.ndarray):
            y_data = torch.from_numpy(y_data)
        
        # Convert to torch tensor
        if isinstance(x_data, np.ndarray):
            x_data = torch.from_numpy(x_data)
        
        # Check samples dimensions.
        # The dimension of the data is expected to be (N, C, L)
        # where N is the #samples, C: #channels, and L is the sequence length
        if len(x_data.shape) == 2:
            x_data = x_data.unsqueeze(1)
        elif len(x_data.shape) == 3 and x_data.shape[1] != self.num_channels:
            x_data = x_data.transpose(1, 2)

        # Normalize data
        if dataset_configs.normalize:
            data_mean = torch.mean(x_data, dim=(0, 2))
            data_std = torch.std(x_data, dim=(0, 2))
            self.transform = transforms.Normalize(mean=data_mean, std=data_std)
        else:
            self.transform = None
        self.x_data = x_data.float()
        self.y_data = y_data.long() if y_data is not None else None
        self.len = x_data.shape[0]
         

    def __getitem__(self, index):
        x = self.x_data[index]
        if self.transform:
            x = self.transform(self.x_data[index].reshape(self.num_channels, -1, 1)).reshape(self.x_data[index].shape)
        y = self.y_data[index] if self.y_data is not None else None
        return x, y

    def __len__(self):
        return self.len


def data_generator(data_path, domain_id, dataset_configs, hparams, dtype):
    # loading dataset file from path
    dataset_file = torch.load(os.path.join(data_path, f"{dtype}_{domain_id}.pt"))

    # Loading datasets
    dataset = Load_Dataset(dataset_file, dataset_configs)

    if dtype == "test":  # you don't need to shuffle or drop last batch while testing
        shuffle  = False
        drop_last = False
    else:
        shuffle = dataset_configs.shuffle
        drop_last = dataset_configs.drop_last

    # Dataloaders
    data_loader = torch.utils.data.DataLoader(dataset=dataset, 
                                              batch_size=hparams["batch_size"],
                                              shuffle=shuffle, 
                                              drop_last=drop_last, 
                                              num_workers=0)
    return data_loader

###############################################################################################################
# for balanced source data 
###############################################################################################################
# total sample number = same / up-sampling / specific value
def data_generator_balanced(data_path, domain_id, dataset_configs, hparams, dtype, total_num_samples='same'):
    # loading dataset file from path
    dataset_file = torch.load(os.path.join(data_path, f"{dtype}_{domain_id}.pt"))

    # Loading datasets
    dataset = Load_Dataset(dataset_file, dataset_configs)

    if dtype == "test":  # you don't need to shuffle or drop last batch while testing
        shuffle  = False
        drop_last = False
    else:
        shuffle = dataset_configs.shuffle
        drop_last = dataset_configs.drop_last

    # sampled  total number 
    num_samples = len(dataset)
    class_counts = dict(Counter(dataset.y_data.tolist()))
    
    if total_num_samples == 'same': total_num_samples = int(num_samples)
    if total_num_samples =='up-sampling': total_num_samples = max(class_counts.values())*len(class_counts)

    # Weights
    labels = dataset.y_data.tolist()
    class_weights = [num_samples / class_counts[i] for i in range(len(class_counts))] 
    weights = [class_weights[labels[i]] for i in range(int(num_samples))]    
    weights = torch.DoubleTensor(weights)                                      
    balanced_sampler = WeightedRandomSampler(torch.DoubleTensor(weights), total_num_samples, replacement=True)
    # Dataloaders
    data_loader = torch.utils.data.DataLoader(dataset=dataset, 
                                              batch_size=hparams["batch_size"],
                                              shuffle=False, 
                                              drop_last=drop_last, 
                                              sampler = balanced_sampler,
                                              num_workers=0)
    return data_loader

###############################################################################################################
# for balanced target
###############################################################################################################
# total sample number = same / up-sampling / specific value
# each_estimatated => each samples' estimated class probabilities / class_weighted = marginal estimation results

def data_generator_plus_estimated_balanced_PB(dataset,dataset_configs,hparams, infer_predictions_results):
    '''
    reference: https://github.com/acmi-lab/RLSbench/blob/main/RLSbench/helper.py#L589
    '''
    # Weights
    num_samples = len(dataset)
    class_num = dataset_configs.num_classes
    class_counts = np.bincount(infer_predictions_results, minlength=class_num)
    max_count = max(class_counts)
    class_counts_proxy = class_counts + 1e-8
    class_weights = max_count / class_counts_proxy
    class_weights[class_counts == 0] = 0
    weights = class_weights[infer_predictions_results]
    balanced_sampler = WeightedRandomSampler(weights, len(weights))

    data_loader = torch.utils.data.DataLoader(dataset=dataset, 
                                              batch_size=hparams["batch_size"],
                                              drop_last = True,
                                              sampler = balanced_sampler,
                                              num_workers=0)
    return data_loader

def data_generator_plus_estimated_balanced_v2(dataset,dataset_configs,hparams,dtype, total_num_samples='same', each_estimatated=None, marginal_class=None, epoch=0):
    each_estimatated  = np.array(each_estimatated)
    infer_predictions_results  = np.argmax(each_estimatated, axis=1)
    # Weights
    num_samples = len(dataset)
    class_num = dataset_configs.num_classes
    class_counts = np.bincount(infer_predictions_results, minlength=class_num)
    max_count = max(class_counts)
    class_counts_proxy = class_counts + 1e-8
    class_weights = max_count / class_counts_proxy
    class_weights[class_counts == 0] = 0
    weights = class_weights[infer_predictions_results]
    balanced_sampler = WeightedRandomSampler(weights, len(weights))

    data_loader = torch.utils.data.DataLoader(dataset=dataset, 
                                              batch_size=hparams["batch_size"],
                                              drop_last = True,
                                              sampler = balanced_sampler,
                                              num_workers=0)
    return data_loader


def data_generator_plus_estimated_balanced(dataset,dataset_configs,hparams,dtype, total_num_samples='same', each_estimatated=None, marginal_class=None, epoch=0):
    if dtype == "test":  # you don't need to shuffle or drop last batch while testing
        shuffle  = False
        drop_last = False
    else:
        shuffle = dataset_configs.shuffle
        drop_last = dataset_configs.drop_last
    
    # Weights
    num_samples = len(dataset)
    class_num = dataset_configs.num_classes
    
    # weight of each data ##############################################################    
    each_estimatated = list(each_estimatated)
    marginal_class = list(marginal_class) #marginal_class.detach().cpu().numpy().tolist() #list(marginal_class.detach())
    weights = list()

    normalized_each_estimated = []
    for estimates in each_estimatated:
        max_value = max(estimates) + 1e-8
        normalized_estimates = [x / max_value for x in estimates]
        normalized_each_estimated.append(normalized_estimates)
    # scaling factor
    smoothing_factor = math.exp(-(1+epoch) / 5) # (init) 5
    class_weights = [(1+0.00001)/(p+0.000001) for p in marginal_class] 
    for i in range(num_samples):
        each_weight = 0
        total_estimated = sum(normalized_each_estimated[i]) 
        for j in range(class_num):
            base_weight = (normalized_each_estimated[i][j] / total_estimated) * class_weights[j]
            smooth_weight = (1.0 / class_num) * smoothing_factor + base_weight * (1 - smoothing_factor)
            each_weight += smooth_weight
        weights.append(each_weight)
    
    ##########################################################################################
    # sampled  total number
    if total_num_samples == 'same': total_num_samples = int(num_samples)
    if total_num_samples =='up-sampling': 
        #print('weights:',weights)
        total_num_samples = int(max(weights))*int(num_samples)*int(class_num)

    balanced_sampler = WeightedRandomSampler(torch.DoubleTensor(weights), total_num_samples, replacement=True)
    # Dataloaders
    data_loader = torch.utils.data.DataLoader(dataset=dataset, 
                                              batch_size=hparams["batch_size"],
                                              shuffle=False, # default
                                              drop_last=drop_last, 
                                              sampler = balanced_sampler,
                                              num_workers=0)
    return data_loader

###############################################################################################################
# for source w.r.t. estimated target marignal dstn
###############################################################################################################
# total sample number = same / up-sampling / specific value
# each_estimatated => each samples' estimated class probabilities / class_weighted = marginal estimation results
def data_generator_plus_estimated_dstn(dataset,dataset_configs,hparams,dtype, total_num_samples='same', marginal_class=False):
    if dtype == "test": 
        shuffle  = False
        drop_last = False
    else:
        shuffle = dataset_configs.shuffle
        drop_last = dataset_configs.drop_last
    
    # Weights
    num_samples = len(dataset)
    class_num = len(dict(Counter(dataset.y_data.tolist())))
    class_weights = list(marginal_class)
    # weight of each data ##############################################################
    labels = dataset.y_data.tolist()
    weights = [class_weights[labels[i]] for i in range(int(num_samples))] 
    ######################################################################################
    
    # sampled  total number
    if total_num_samples == 'same': total_num_samples = int(num_samples)
    if total_num_samples =='up-sampling': total_num_samples = round(max(marginal_class)*num_samples)*class_num  
        
    balanced_sampler = WeightedRandomSampler(torch.DoubleTensor(weights), total_num_samples, replacement=True)
    # Dataloaders
    data_loader = torch.utils.data.DataLoader(dataset=dataset, 
                                              batch_size=hparams["batch_size"],
                                              shuffle=False, 
                                              drop_last=drop_last, 
                                              sampler = balanced_sampler,
                                              num_workers=0)
    return data_loader
###############################################################################################################



def data_generator_old(data_path, domain_id, dataset_configs, hparams):
    # loading path
    train_dataset = torch.load(os.path.join(data_path, "train_" + domain_id + ".pt"))
    test_dataset = torch.load(os.path.join(data_path, "test_" + domain_id + ".pt"))

    # Loading datasets
    train_dataset = Load_Dataset(train_dataset, dataset_configs)
    test_dataset = Load_Dataset(test_dataset, dataset_configs)

    # Dataloaders
    batch_size = hparams["batch_size"]
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
                                               shuffle=True, drop_last=True, num_workers=0)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size,
                                              shuffle=False, drop_last=dataset_configs.drop_last, num_workers=0)
    return train_loader, test_loader



def few_shot_data_generator(data_loader, dataset_configs, num_samples=5):
    x_data = data_loader.dataset.x_data
    y_data = data_loader.dataset.y_data

    NUM_SAMPLES_PER_CLASS = num_samples
    NUM_CLASSES = len(torch.unique(y_data))

    counts = [y_data.eq(i).sum().item() for i in range(NUM_CLASSES)]
    samples_count_dict = {i: min(counts[i], NUM_SAMPLES_PER_CLASS) for i in range(NUM_CLASSES)}

    samples_ids = {i: torch.where(y_data == i)[0] for i in range(NUM_CLASSES)}
    selected_ids = {i: torch.randperm(samples_ids[i].size(0))[:samples_count_dict[i]] for i in range(NUM_CLASSES)}

    selected_x = torch.cat([x_data[samples_ids[i][selected_ids[i]]] for i in range(NUM_CLASSES)], dim=0)
    selected_y = torch.cat([y_data[samples_ids[i][selected_ids[i]]] for i in range(NUM_CLASSES)], dim=0)

    few_shot_dataset = {"samples": selected_x, "labels": selected_y}
    few_shot_dataset = Load_Dataset(few_shot_dataset, dataset_configs)

    few_shot_loader = torch.utils.data.DataLoader(dataset=few_shot_dataset, batch_size=len(few_shot_dataset),
                                                  shuffle=False, drop_last=False, num_workers=0)

    return few_shot_loader

