import torch
import torch.nn as nn
import numpy as np
import itertools    

from models.models import classifier, ReverseLayerF, Discriminator, RandomLayer, Discriminator_CDAN, \
    codats_classifier, AdvSKM_Disc, CNN_ATTN
from models.loss import MMD_loss, CORAL, ConditionalEntropyLoss, VAT, LMMD_loss, HoMM_loss, NTXentLoss, SupConLoss
from utils import EMA
from torch.optim.lr_scheduler import StepLR
from copy import deepcopy
import torch.nn. functional as F
from pytorch_metric_learning import losses

## additional
from models.models import tf_encoder, tf_decoder, CNN_Decoder, TCN_Decoder
from models.loss import SinkhornDistance
from collections import Counter
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from dataloader.dataloader import data_generator_plus_estimated_balanced_PB, data_generator_plus_estimated_balanced,data_generator_plus_estimated_dstn
from utils import AverageMeter
import collections

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import math

from torchmetrics import Accuracy, F1Score

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import jensenshannon


def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""  
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]

# Updated decomposed_features function
from scipy.signal import cwt, ricker
from statsmodels.nonparametric.smoothers_lowess import lowess

def moving_average(x, window_size):
    return x.rolling(window=window_size, min_periods=1, center=True).mean()

def decomposed_features(x, window_size=5, wavelet_width=5, frac=0.1): 
    x = x.numpy()
    x = x.reshape(x.shape[1], -1).T
    x_df = pd.DataFrame(x)
    trend_list = []
    de_trend_list = []
    frequency_list = []
    
    for col in x_df.columns:
        x_series = x_df[col]
        
        # Calculate non-linear trend using Lowess
        trend = lowess(x_series, np.arange(len(x_series)), frac=frac)[:, 1]
        trend_list.append(trend)
        
        # Calculate de-trend component as the residual
        de_trend = x_series - trend
        de_trend_list.append(de_trend.values)
        
        # Frequency part (FFT)
        frequency = abs(np.fft.fft(x_series))  # magnitudes
        frequency_list.append(frequency)
    
    # Convert results to numpy arrays
    trend_array = np.array(trend_list)
    de_trend_array = np.array(de_trend_list)
    frequency_array = np.array(frequency_list)
    
    return trend_array, de_trend_array, frequency_array
    
########################################################################################
class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a domain adaptation algorithm.
    Subclasses should implement the update() method.
    """

    def __init__(self, configs, backbone):
        super(Algorithm, self).__init__()
        self.configs = configs
        self.cross_entropy = nn.CrossEntropyLoss()
        self.feature_extractor = backbone(configs)
        self.classifier = classifier(configs)
        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        # metrics
        self.num_classes = self.configs.num_classes
        self.ACC = Accuracy(task="multiclass", num_classes=self.num_classes).cpu()
        self.F1 = F1Score(task="multiclass", num_classes=self.num_classes, average="macro").cpu()

    # update function is common to all algorithms
    def update(self, src_loader, trg_loader, avg_meter, logger):
        # defining best and last model
        best_src_risk = float('inf')
        best_model = None

        for epoch in range(1, self.hparams["num_epochs"] + 1):
            # training loop 
            self.training_epoch(src_loader, trg_loader, avg_meter, epoch)

            # saving the best model based on src risk
            if (epoch + 1) % 10 == 0 and avg_meter['Src_cls_loss'].avg < best_src_risk:
                best_src_risk = avg_meter['Src_cls_loss'].avg
                best_model = deepcopy(self.network.state_dict())

            logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')
        
        last_model = self.network.state_dict()

        return last_model, best_model

    def update_label_shift(self, args, src_loader, trg_loader, src_balanced_loader, trg_train_b1_dl,
                           avg_meter, logger, hparams, device, dataset_configs, warm_up_align=None,FE_balanced=False,src_train_b1_dl=None,trg_test_b1_dl=None):
        # defining best and last model
        best_src_risk = float('inf')
        best_model = None
        # hparams
        self.hparams = hparams
        self.device = device
        # avg_meters_warm
        avg_meters_warm = collections.defaultdict(lambda: AverageMeter())

        # Extract Target Only information
        num_classes = dataset_configs.num_classes

        with torch.no_grad():
            self.feature_extractor.eval()
            # target
            trg_feat_trend = list()
            trg_feat_detrend = list()
            trg_feat_frequency = list()
            trg_pseudo_labels = []
            for trg_x, _ in trg_train_b1_dl:
                trg_x = trg_x.to(device)
                # Extract trend, detrend, frequency, and wavelet features for co-training clustering
                trend_array, detrend_array, frequency_array = decomposed_features(trg_x.cpu(), window_size=5, wavelet_width=5, frac=float(args.frac))
                trg_feat_trend.append(trend_array)
                trg_feat_detrend.append(detrend_array)
                trg_feat_frequency.append(frequency_array)
            
            # Feature sets for experiments: Select which feature to use
            feature_sets = []
            feature_sets = [('trend', trg_feat_trend),('detrend', trg_feat_detrend),('frequency', trg_feat_frequency)]

            # Clustering using different feature sets
            clustering_method = args.clustering_method if hasattr(args, 'clustering_method') else 'gmm'
            num_samples = len(trg_feat_frequency)
            n_components = num_classes
            
            cluster_labels_list = []
            cluster_probabilities_list = []
            cluster_algorithm = []
    
            for feature_name, trg_feat_list in feature_sets:
                if len(trg_feat_list) == 0:
                    logger.warning(f"Feature list {feature_name} is empty. Skipping clustering for this feature.")
                    continue
                clustered_X = np.array(trg_feat_list).reshape(len(trg_feat_list), -1)
                trg_X = np.array(trg_feat_list).reshape(len(trg_feat_list), -1)
                
                if clustering_method == 'gmm':
                    cluster_name = GaussianMixture(n_components=n_components, random_state=0)
                    cluster_name.fit(clustered_X)
                    cluster_labels = cluster_name.predict(trg_X)
                    cluster_probabilities = cluster_name.predict_proba(trg_X)
                cluster_algorithm.append(cluster_name)
                
                cluster_labels_list.append(cluster_labels)
                cluster_probabilities_list.append(cluster_probabilities)
            
            if len(feature_sets) > 1:
                # Calculate co-association matrix if more than one feature set is used
                co_association_matrix = np.zeros((num_samples, num_samples))
                for labels in cluster_labels_list:
                    for i in range(num_samples):
                        for j in range(num_samples):
                            if labels[i] == labels[j]:
                                co_association_matrix[i, j] += 1
                co_association_matrix /= len(cluster_labels_list)

                # Set diagonal elements to zero
                np.fill_diagonal(co_association_matrix, 0)
                
                agg_clustering = AgglomerativeClustering(n_clusters=n_components, linkage='average')
                final_cluster_labels = agg_clustering.fit_predict(1 - co_association_matrix)                
                final_cluster_probabilities = np.zeros((num_samples, n_components))
                for i in range(num_samples):
                    for j in range(num_samples):
                        final_cluster_probabilities[i, final_cluster_labels[j]] += co_association_matrix[i, j]
                
                # Normalize the probabilities
                final_cluster_probabilities /= final_cluster_probabilities.sum(axis=1, keepdims=True)
                
                # Store the results
                trg_clustered_intensity_list = [(final_cluster_labels[idx], final_cluster_probabilities[idx]) for idx in range(num_samples)]
                cluster_algorithm = (cluster_algorithm, agg_clustering)
        
        for epoch in range(1, self.hparams["num_epochs"] + 1):
            self.feature_extractor.train()
            self.classifier.train()
            self.training_epoch(src_balanced_loader, trg_loader, avg_meter, epoch)

            # saving the best model based on src risk
            if (epoch + 1) % 10 == 0 and avg_meter['Src_cls_loss'].avg < best_src_risk:
                best_src_risk = avg_meter['Src_cls_loss'].avg
                best_model = deepcopy(self.network.state_dict())

            logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')
                      
        last_model = self.network.state_dict()
        return last_model, best_model, None, cluster_algorithm


class DANN(Algorithm):
    """
    DANN: https://arxiv.org/abs/1505.07818
    """

    def __init__(self, backbone, configs, hparams, device):
        super().__init__(configs, backbone)

        
        # optimizer and scheduler
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        # hparams
        self.hparams = hparams
        # device
        self.device = device

        # Domain Discriminator
        self.domain_classifier = Discriminator(configs)
        self.optimizer_disc = torch.optim.Adam(
            self.domain_classifier.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"], betas=(0.5, 0.99)
        )
        # backbone
        self.backbone = configs.backbone
        

    def training_epoch(self,src_loader, trg_loader, avg_meter, epoch):
        # Combine dataloaders
        # Method 1 (min len of both domains)
        # joint_loader = enumerate(zip(src_loader, trg_loader))

        # Method 2 (max len of both domains)
        # joint_loader =enumerate(zip(src_loader, itertools.cycle(trg_loader)))
        joint_loader =enumerate(zip(src_loader, itertools.cycle(trg_loader)))
        num_batches = max(len(src_loader), len(trg_loader))

        for step, ((src_x, src_y), (trg_x, _)) in joint_loader:

            src_x, src_y, trg_x = src_x.to(self.device), src_y.to(self.device), trg_x.to(self.device)
            
            p = float(step + epoch * num_batches) / self.hparams["num_epochs"] + 1 / num_batches
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            # zero grad
            self.optimizer.zero_grad()
            self.optimizer_disc.zero_grad()

            domain_label_src = torch.ones(len(src_x)).to(self.device)
            domain_label_trg = torch.zeros(len(trg_x)).to(self.device)

            if self.backbone=="tf_encoder": src_feat, _ = self.feature_extractor(src_x)
            else: src_feat = self.feature_extractor(src_x)
            src_pred = self.classifier(src_feat)

            if self.backbone=="tf_encoder": trg_feat, _ = self.feature_extractor(trg_x)
            else: trg_feat = self.feature_extractor(trg_x)

            # Task classification  Loss
            src_cls_loss = self.cross_entropy(src_pred.squeeze(), src_y)

            # Domain classification loss
            # source
            src_feat_reversed = ReverseLayerF.apply(src_feat, alpha)
            src_domain_pred = self.domain_classifier(src_feat_reversed)
            src_domain_loss = self.cross_entropy(src_domain_pred, domain_label_src.long())

            # target
            trg_feat_reversed = ReverseLayerF.apply(trg_feat, alpha)
            trg_domain_pred = self.domain_classifier(trg_feat_reversed)
            trg_domain_loss = self.cross_entropy(trg_domain_pred, domain_label_trg.long())

            # Total domain loss
            domain_loss = src_domain_loss + trg_domain_loss

            loss = self.hparams["src_cls_loss_wt"] * src_cls_loss + \
                self.hparams["domain_loss_wt"] * domain_loss

            loss.backward()
            self.optimizer.step()
            self.optimizer_disc.step()

            losses =  {'Total_loss': loss.item(), 'Domain_loss': domain_loss.item(), 'Src_cls_loss': src_cls_loss.item()}
           
            for key, val in losses.items():
                avg_meter[key].update(val, 32)

        self.lr_scheduler.step()



class DIRT(Algorithm):
    """
    DIRT-T: https://arxiv.org/abs/1802.08735
    """

    def __init__(self, backbone, configs, hparams, device):
        super().__init__(configs, backbone)

        # optimizer and scheduler
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        # hparams
        self.hparams = hparams
        # device
        self.device = device


        # Aligment losses
        self.criterion_cond = ConditionalEntropyLoss().to(device)
        self.vat_loss = VAT(self.feature_extractor,self.classifier, configs, device).to(device)
        self.ema = EMA(0.998)
        self.ema.register(self.network)

        # Discriminator
        self.domain_classifier = Discriminator(configs)
        self.optimizer_disc = torch.optim.Adam(
            self.domain_classifier.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        # backbone
        self.backbone = configs.backbone
       
    def training_epoch(self,src_loader, trg_loader, avg_meter, epoch):

        # Construct Joint Loaders 
        joint_loader =enumerate(zip(src_loader, itertools.cycle(trg_loader)))

        for step, ((src_x, src_y), (trg_x, _)) in joint_loader:
            src_x, src_y, trg_x = src_x.to(self.device), src_y.to(self.device), trg_x.to(self.device)
            # prepare true domain labels
            domain_label_src = torch.ones(len(src_x)).to(self.device)
            domain_label_trg = torch.zeros(len(trg_x)).to(self.device)
            domain_label_concat = torch.cat((domain_label_src, domain_label_trg), 0).long()

            if self.backbone=="tf_encoder": src_feat, _ = self.feature_extractor(src_x)
            else: src_feat = self.feature_extractor(src_x)
            src_pred = self.classifier(src_feat)

            # target features and predictions
            if self.backbone=="tf_encoder": trg_feat, _ = self.feature_extractor(trg_x)
            else: trg_feat = self.feature_extractor(trg_x)
            trg_pred = self.classifier(trg_feat)

            # concatenate features and predictions
            feat_concat = torch.cat((src_feat, trg_feat), dim=0)

            # Domain classification loss
            disc_prediction = self.domain_classifier(feat_concat.detach())
            disc_loss = self.cross_entropy(disc_prediction, domain_label_concat)

            # update Domain classification
            self.optimizer_disc.zero_grad()
            disc_loss.backward()
            self.optimizer_disc.step()

            # prepare fake domain labels for training the feature extractor
            domain_label_src = torch.zeros(len(src_x)).long().to(self.device)
            domain_label_trg = torch.ones(len(trg_x)).long().to(self.device)
            domain_label_concat = torch.cat((domain_label_src, domain_label_trg), 0)

            # Repeat predictions after updating discriminator
            disc_prediction = self.domain_classifier(feat_concat)

            # loss of domain discriminator according to fake labels
            domain_loss = self.cross_entropy(disc_prediction, domain_label_concat)

            # Task classification  Loss
            src_cls_loss = self.cross_entropy(src_pred.squeeze(), src_y)

            # conditional entropy loss.
            loss_trg_cent = self.criterion_cond(trg_pred)

            # Virual advariarial training loss
            loss_src_vat = self.vat_loss(src_x, src_pred)
            loss_trg_vat = self.vat_loss(trg_x, trg_pred)
            total_vat = loss_src_vat + loss_trg_vat
            # total loss
            loss = self.hparams["src_cls_loss_wt"] * src_cls_loss + self.hparams["domain_loss_wt"] * domain_loss + \
                self.hparams["cond_ent_wt"] * loss_trg_cent + self.hparams["vat_loss_wt"] * total_vat

            # update exponential moving average
            self.ema(self.network)

            # update feature extractor
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses =  {'Total_loss': loss.item(), 'Domain_loss': domain_loss.item(), 'Src_cls_loss': src_cls_loss.item(),
                    'cond_ent_loss': loss_trg_cent.item()}

            for key, val in losses.items():
                avg_meter[key].update(val, 32)

        self.lr_scheduler.step()

class DSAN(Algorithm):
    """
    DSAN: https://ieeexplore.ieee.org/document/9085896
    """

    def __init__(self, backbone, configs, hparams, device):
        super().__init__(configs, backbone)

        # optimizer and scheduler
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        # hparams
        self.hparams = hparams
        # device
        self.device = device

        # Alignment losses
        self.loss_LMMD = LMMD_loss(device=device, class_num=configs.num_classes).to(device)
        # backbone
        self.backbone = configs.backbone

    def training_epoch(self,src_loader, trg_loader, avg_meter, epoch):

        # Construct Joint Loaders 
        joint_loader =enumerate(zip(src_loader, itertools.cycle(trg_loader)))

        for step, ((src_x, src_y), (trg_x, _)) in joint_loader:
            src_x, src_y, trg_x = src_x.to(self.device), src_y.to(self.device), trg_x.to(self.device)        # extract source features
            
            if self.backbone=="tf_encoder": 
                src_feat, _ = self.feature_extractor(src_x)
            else: 
                src_feat = self.feature_extractor(src_x)
            src_pred = self.classifier(src_feat)

            # extract target features
            if self.backbone=="tf_encoder": 
                trg_feat, _ = self.feature_extractor(trg_x)
            else: 
                trg_feat = self.feature_extractor(trg_x)
            trg_pred = self.classifier(trg_feat)
            
            # calculate lmmd loss
            domain_loss = self.loss_LMMD.get_loss(src_feat, trg_feat, src_y, torch.nn.functional.softmax(trg_pred, dim=1))

            # calculate source classification loss
            src_cls_loss = self.cross_entropy(src_pred, src_y)

            # calculate the total loss
            loss = self.hparams["domain_loss_wt"] * domain_loss + \
                self.hparams["src_cls_loss_wt"] * src_cls_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses =  {'Total_loss': loss.item(), 'LMMD_loss': domain_loss.item(), 'Src_cls_loss': src_cls_loss.item()}

            for key, val in losses.items():
                avg_meter[key].update(val, 32)

        self.lr_scheduler.step()

class HoMM(Algorithm):
    """
    HoMM: https://arxiv.org/pdf/1912.11976.pdf
    """

    def __init__(self, backbone, configs, hparams, device):
        super().__init__(configs, backbone)

        # optimizer and scheduler
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        # hparams
        self.hparams = hparams
        # device
        self.device = device

        # aligment losses
        self.coral = CORAL()
        self.HoMM_loss = HoMM_loss()
        # backbone
        self.backbone = configs.backbone

    def training_epoch(self,src_loader, trg_loader, avg_meter, epoch):

        # Construct Joint Loaders 
        joint_loader =enumerate(zip(src_loader, itertools.cycle(trg_loader)))

        for step, ((src_x, src_y), (trg_x, _)) in joint_loader:
            src_x, src_y, trg_x = src_x.to(self.device), src_y.to(self.device), trg_x.to(self.device)           # extract source features
            
            if self.backbone=="tf_encoder": src_feat, _ = self.feature_extractor(src_x)
            else: src_feat = self.feature_extractor(src_x)
            src_pred = self.classifier(src_feat)

            # extract target features
            if self.backbone=="tf_encoder": trg_feat, _ = self.feature_extractor(trg_x)
            else: trg_feat = self.feature_extractor(trg_x)
            trg_pred = self.classifier(trg_feat)

            # calculate source classification loss
            src_cls_loss = self.cross_entropy(src_pred, src_y)

            # calculate lmmd loss
            domain_loss = self.HoMM_loss(src_feat, trg_feat)

            # calculate the total loss
            loss = self.hparams["domain_loss_wt"] * domain_loss + \
                self.hparams["src_cls_loss_wt"] * src_cls_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses =  {'Total_loss': loss.item(), 'HoMM_loss': domain_loss.item(), 'Src_cls_loss': src_cls_loss.item()}
            
            for key, val in losses.items():
                avg_meter[key].update(val, 32)

        self.lr_scheduler.step()




class CoDATS(Algorithm):
    """
    CoDATS: https://arxiv.org/pdf/2005.10996.pdf
    """

    def __init__(self, backbone, configs, hparams, device):
        super().__init__(configs, backbone)

        # we replace the original classifier with codats the classifier
        # remember to use same name of self.classifier, as we use it for the model evaluation
        self.classifier = codats_classifier(configs)
        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        # optimizer and scheduler
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        # hparams
        self.hparams = hparams
        # device
        self.device = device


        # Domain classifier
        self.domain_classifier = Discriminator(configs)

        self.optimizer_disc = torch.optim.Adam(
            self.domain_classifier.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"], betas=(0.5, 0.99)
        )
        # backbone
        self.backbone = configs.backbone

    def training_epoch(self,src_loader, trg_loader, avg_meter, epoch):

        # Construct Joint Loaders 
        joint_loader =enumerate(zip(src_loader, itertools.cycle(trg_loader)))
        num_batches = max(len(src_loader), len(trg_loader))
        for step, ((src_x, src_y), (trg_x, _)) in joint_loader:
            src_x, src_y, trg_x = src_x.to(self.device), src_y.to(self.device), trg_x.to(self.device)           # extract source features
        
            p = float(step + epoch * num_batches) / self.hparams["num_epochs"] + 1 / num_batches
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            # zero grad
            self.optimizer.zero_grad()
            self.optimizer_disc.zero_grad()

            domain_label_src = torch.ones(len(src_x)).to(self.device)
            domain_label_trg = torch.zeros(len(trg_x)).to(self.device)

            if self.backbone=="tf_encoder": src_feat, _ = self.feature_extractor(src_x)
            else: src_feat = self.feature_extractor(src_x)
            src_pred = self.classifier(src_feat)

            if self.backbone=="tf_encoder": trg_feat, _ = self.feature_extractor(trg_x)
            else: trg_feat = self.feature_extractor(trg_x)

            # Task classification  Loss
            src_cls_loss = self.cross_entropy(src_pred.squeeze(), src_y)

            # Domain classification loss
            # source
            src_feat_reversed = ReverseLayerF.apply(src_feat, alpha)
            src_domain_pred = self.domain_classifier(src_feat_reversed)
            src_domain_loss = self.cross_entropy(src_domain_pred, domain_label_src.long())

            # target
            trg_feat_reversed = ReverseLayerF.apply(trg_feat, alpha)
            trg_domain_pred = self.domain_classifier(trg_feat_reversed)
            trg_domain_loss = self.cross_entropy(trg_domain_pred, domain_label_trg.long())

            # Total domain loss
            domain_loss = src_domain_loss + trg_domain_loss

            loss = self.hparams["src_cls_loss_wt"] * src_cls_loss + \
                self.hparams["domain_loss_wt"] * domain_loss

            loss.backward()
            self.optimizer.step()
            self.optimizer_disc.step()

            losses =  {'Total_loss': loss.item(), 'Domain_loss': domain_loss.item(), 'Src_cls_loss': src_cls_loss.item()}
            for key, val in losses.items():
                avg_meter[key].update(val, 32)

        self.lr_scheduler.step()


class SASA(Algorithm):
    
    def __init__(self, backbone, configs, hparams, device):
        super().__init__(configs, backbone)

        # feature_length for classifier
        configs.features_len = 1
        self.classifier = classifier(configs)
        # feature length for feature extractor
        configs.features_len = 1
        self.feature_extractor = CNN_ATTN(configs)
        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        # optimizer and scheduler
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        # hparams
        self.hparams = hparams
        # device
        self.device = device
        # backbone
        self.backbone = configs.backbone


    def training_epoch(self,src_loader, trg_loader, avg_meter, epoch):

        # Construct Joint Loaders 
        joint_loader =enumerate(zip(src_loader, itertools.cycle(trg_loader)))
        for step, ((src_x, src_y), (trg_x, _)) in joint_loader:
            src_x, src_y, trg_x = src_x.to(self.device), src_y.to(self.device), trg_x.to(self.device)         # extract source features

            # Extract features
            if self.backbone=="tf_encoder": src_feature, _ = self.feature_extractor(src_x)
            else: src_feature = self.feature_extractor(src_x)
            if self.backbone=="tf_encoder": tgt_feature, _ = self.feature_extractor(trg_x)
            else: tgt_feature = self.feature_extractor(trg_x)
                
            # source classification loss
            y_pred = self.classifier(src_feature)
            src_cls_loss = self.cross_entropy(y_pred, src_y)

            # MMD loss
            domain_loss_intra = self.mmd_loss(src_struct=src_feature,
                                            tgt_struct=tgt_feature, weight=self.hparams['domain_loss_wt'])

            # total loss
            total_loss = self.hparams['src_cls_loss_wt'] * src_cls_loss + domain_loss_intra

            # remove old gradients
            self.optimizer.zero_grad()
            # calculate gradients
            total_loss.backward()
            # update the weights
            self.optimizer.step()

            losses =  {'Total_loss': total_loss.item(), 'MMD_loss': domain_loss_intra.item(),
                    'Src_cls_loss': src_cls_loss.item()}
            for key, val in losses.items():
                avg_meter[key].update(val, 32)

        self.lr_scheduler.step()
    def mmd_loss(self, src_struct, tgt_struct, weight):
        delta = torch.mean(src_struct - tgt_struct, dim=-2)
        loss_value = torch.norm(delta, 2) * weight
        return loss_value

