import sys
sys.path.append('../../ADATIME/')
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy, AUROC, F1Score
import os
import wandb
import pandas as pd
import numpy as np
import warnings
import sklearn.exceptions
import collections

from torchmetrics import Accuracy, AUROC, F1Score
from dataloader.dataloader import data_generator, few_shot_data_generator, data_generator_balanced, data_generator_original
from configs.data_model_configs import get_dataset_class
from configs.hparams import get_hparams_class
from configs.sweep_params import sweep_alg_hparams
from utils import fix_randomness, starting_logs, DictAsObject,AverageMeter
from algorithms.algorithms import get_algorithm_class
from models.models import get_backbone_class

warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)

# additional 
from dataloader.dataloader import data_generator_balanced
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import math


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
    

class AbstractTrainer(object):
    """
   This class contain the main training functions for our AdAtime
    """

    def __init__(self, args):
        self.da_method = args.da_method  # Selected  DA Method
        self.dataset = args.dataset  # Selected  Dataset
        self.backbone = args.backbone
        self.device = torch.device(args.device)  # device

        # Exp Description
        self.experiment_description = args.dataset 
        self.run_description = f"{args.da_method}_{args.exp_name}"
        
        # paths
        self.home_path =  os.getcwd() #os.path.dirname(os.getcwd())
        self.save_dir = args.save_dir
        self.data_path = os.path.join(args.data_path, self.dataset)

        self.create_save_dir(os.path.join(self.home_path,  self.save_dir ))
        self.exp_log_dir = os.path.join(self.home_path, self.save_dir, self.experiment_description, f"{self.run_description}")
        os.makedirs(self.exp_log_dir, exist_ok=True)

        # Specify runs
        self.num_runs = args.num_runs

        # get dataset and base model configs
        self.dataset_configs, self.hparams_class = self.get_configs()
        self.dataset_configs.backbone = args.backbone

        # to fix dimension of features in classifier and discriminator networks.
        self.dataset_configs.final_out_channels = self.dataset_configs.tcn_final_out_channles if args.backbone == "TCN" else self.dataset_configs.final_out_channels

        # Specify number of hparams
        self.hparams = {**self.hparams_class.alg_hparams[self.da_method],
                                **self.hparams_class.train_params}

        # metrics
        self.num_classes = self.dataset_configs.num_classes
        self.ACC = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.F1 = F1Score(task="multiclass", num_classes=self.num_classes, average="macro")
        self.AUROC = AUROC(task="multiclass", num_classes=self.num_classes)        

        # metrics

    def sweep(self):
        # sweep configurations
        pass
    
    def initialize_algorithm(self):
        # get algorithm class
        algorithm_class = get_algorithm_class(self.da_method)
        backbone_fe = get_backbone_class(self.backbone)

        # Initilaize the algorithm
        self.algorithm = algorithm_class(backbone_fe, self.dataset_configs, self.hparams, self.device)
        self.algorithm.to(self.device)

    def load_checkpoint(self, model_dir):
        checkpoint = torch.load(os.path.join(self.home_path, model_dir, 'checkpoint.pt'))
        last_model = checkpoint['last']
        best_model = checkpoint['best']
        return last_model, best_model

    def train_model(self):
        # Get the algorithm and the backbone network
        algorithm_class = get_algorithm_class(self.da_method)
        backbone_fe = get_backbone_class(self.backbone)

        # Initilaize the algorithm
        self.algorithm = algorithm_class(backbone_fe, self.dataset_configs, self.hparams, self.device)
        self.algorithm.to(self.device)

        # Training the model
        self.last_model, self.best_model = self.algorithm.update(self.src_train_dl, self.trg_train_dl, self.loss_avg_meters, self.logger)
        return self.last_model, self.best_model

    def evaluate(self, mode, test_loader, label_weight=None, cluster_algorithm=None,trg_test_b1_dl=None, logger=None,reweight=False,refinement=True): 
        args = self.args
        if refinement == True:
            if logger!=None: logger.debug('<test refinement>')
            with torch.no_grad():
                # target
                trg_feat_trend = list()
                trg_feat_detrend = list()
                trg_feat_frequency = list()
                # source
                src_feat_trend = list()
                src_feat_detrend = list()
                src_feat_frequency = list()
                trg_pseudo_labels = []
                for trg_x, _ in trg_test_b1_dl:
                    trend_array, detrend_array, frequency_array = decomposed_features(trg_x, window_size=5, wavelet_width=5, frac=float(args.frac))
                    trg_feat_trend.append(trend_array)
                    trg_feat_detrend.append(detrend_array)
                    trg_feat_frequency.append(frequency_array)
                selected_feature = args.selected_feature if hasattr(args, 'selected_feature') else 'all'
                clustering_method = args.clustering_method if hasattr(args, 'clustering_method') else 'gmm'
                feature_sets = []
                feature_sets = [('trend', trg_feat_trend,src_feat_trend),('detrend', trg_feat_detrend,src_feat_detrend),('frequency', trg_feat_frequency,src_feat_frequency)]
                num_samples = len(trg_feat_frequency)
                cluster_labels_list = []
                cluster_probabilities_list = []

                clustering_list, agg_clustering = cluster_algorithm
                n_components = self.num_classes
                for cluster_name,(feature_name, trg_feat_list,src_feat_list) in zip(clustering_list, feature_sets):
                    trg_X = np.array(trg_feat_list).reshape(len(trg_feat_list), -1)
                    cluster_labels = cluster_name.predict(trg_X)
                    cluster_labels_list.append(cluster_labels)

                if len(feature_sets) > 1:
                    co_association_matrix = np.zeros((num_samples, num_samples))
                    for labels in cluster_labels_list:
                        for i in range(num_samples):
                            for j in range(num_samples):
                                if labels[i] == labels[j]:
                                    co_association_matrix[i, j] += 1
                    co_association_matrix /= len(cluster_labels_list)
    
                    # Set diagonal elements to zero
                    np.fill_diagonal(co_association_matrix, 0)
                    
                    # Apply Agglomerative Clustering on co-association matrix
                    agg_clustering = AgglomerativeClustering(n_clusters=n_components, linkage='average', metric=args.metric)
                    final_cluster_labels = agg_clustering.fit_predict(1 - co_association_matrix)
                    #final_cluster_labels = agg_clustering.predict(1 - co_association_matrix)
                    
                    final_cluster_probabilities = np.zeros((num_samples, n_components))
                    for i in range(num_samples):
                        for j in range(num_samples):
                            final_cluster_probabilities[i, final_cluster_labels[j]] += co_association_matrix[i, j]                    
                    final_cluster_probabilities /= final_cluster_probabilities.sum(axis=1, keepdims=True)
                    trg_clustered_intensity_list = [(final_cluster_labels[idx], final_cluster_probabilities[idx]) for idx in range(num_samples)]

        if mode=='trg':
            feature_extractor = self.algorithm.feature_extractor.to(self.device)
            classifier = self.algorithm.classifier.to(self.device)
    
            feature_extractor.eval()
            classifier.eval()
    
            total_loss, preds_list, labels_list = [], [], []
            
            with torch.no_grad():
                trg_confidence_list = []
                trg_refinement_list = []
                full_labels_list = []
                trg_feat_list = []
                for data, labels in test_loader:
                    full_labels_list = full_labels_list + labels.detach().tolist()
                    data = data.float().to(self.device)
                    labels = labels.view((-1)).long().to(self.device)
                    features = feature_extractor(data)                    
                    trg_confidence = classifier(features)
                    trg_confidence_list.append(trg_confidence)
                    labels_list.append(labels)
                    trg_confidence = F.softmax(trg_confidence, dim=1)
                    trg_refinement_list = trg_refinement_list + trg_confidence.detach().tolist()
                    trg_feat = features.reshape(features.shape[0],-1)
                    trg_feat_list = trg_feat_list + trg_feat.detach().tolist()
                pred_class_list = np.argmax(trg_refinement_list,axis=1)
                if refinement==True:
                    class_cluster_dict = dict() # key: class_id, value: cluster_id
                    num_classes = self.num_classes
                    for class_id in range(num_classes):
                        class_indices = [index for index, value in enumerate(pred_class_list) if value == class_id]
                        cluster_id_list =  [trg_clustered_intensity_list[i][0] for i in class_indices]
                        clustered_indensity_list =  [trg_clustered_intensity_list[i][1][trg_clustered_intensity_list[i][0]] for i in class_indices] 
                        class_cluster_indensity_information = list() 
                        for cluster_id in range(num_classes):
                            cluster_indices = [index for index, value in enumerate(cluster_id_list) if value == cluster_id] 
                            class_cluster_indensity_list = [trg_clustered_intensity_list[i][1][trg_clustered_intensity_list[i][0]] for i in cluster_indices] 
                            class_cluster_indensity_information.append(len(class_cluster_indensity_list)) # one value
                        final_cluster_id = np.argmax(class_cluster_indensity_information) # 1dim
                        class_cluster_dict[class_id] = final_cluster_id
        
                    refined_candidate_trg_idx_list = list()
                    for idx in range(len(trg_refinement_list)):
                        pred_class = pred_class_list[idx]
                        pred_class_cluster_idx = class_cluster_dict[pred_class]
                        cluster_idx = trg_clustered_intensity_list[idx][0]
                        if pred_class_cluster_idx!=cluster_idx:
                            refined_candidate_trg_idx_list.append(idx)
                    
                    final_trg_refinement_idx = []
                    for refined_idx in refined_candidate_trg_idx_list:
                        pred_class = pred_class_list[idx]
                        trg_confidence = trg_refinement_list[refined_idx] 
                        clustered_intensity_probs = trg_clustered_intensity_list[refined_idx][1] 

                        refined_trg_confidence = []
                        for class_idx in range(num_classes):
                            most_cluster = class_cluster_dict[class_idx]
                            most_cluster_intensity_probs = clustered_intensity_probs[most_cluster]
                            each_refined_trg_confidence = trg_confidence[class_idx]*((most_cluster_intensity_probs+0.0001)/((1-most_cluster_intensity_probs+0.0001)))
                            refined_trg_confidence.append(each_refined_trg_confidence)
        
                        refined_class = np.argmax(refined_trg_confidence)
                        if refined_class!=pred_class:
                            refined_trg_confidence = [one_conf/sum(refined_trg_confidence) for one_conf in refined_trg_confidence]
                            trg_refinement_list[refined_idx] = refined_trg_confidence
                            final_trg_refinement_idx.append(refined_idx)

                    total_pseudo_labels_per_class = np.zeros([self.num_classes])
                    for trg_confidence in trg_refinement_list:  
                        trg_confidence = np.array(trg_confidence)
                        pseudo_label = np.argmax(trg_confidence, axis=0)
                        total_pseudo_labels_per_class[pseudo_label] += 1
                    total_pseudo_labels_per_class = total_pseudo_labels_per_class / sum(total_pseudo_labels_per_class)

                trg_confidence_list = trg_refinement_list
                
                for trg_confidence,labels in zip(trg_confidence_list,labels_list):
                    if refinement==True:
                        trg_confidence =  torch.tensor(np.array(trg_confidence).reshape((1,-1))).to(self.device) # (c.f) batch_size =1
                        trg_confidence = F.softmax(trg_confidence, dim=1)
                    predictions = trg_confidence
                    # compute loss
                    loss = F.cross_entropy(predictions, labels)
                    total_loss.append(loss.item())
                    pred = predictions.detach()  
    
                    # append predictions and labels
                    preds_list.append(pred)

            self.loss = torch.tensor(total_loss).mean()  # average loss
            self.full_preds = torch.cat((preds_list))
            self.full_labels = torch.cat((labels_list))
        else:
            # mode == ['src','few']
            self.evaluate_src_few(test_loader)
    
    #######################################################################################################################################
        
    def evaluate_src_few(self, test_loader):
        feature_extractor = self.algorithm.feature_extractor.to(self.device)
        classifier = self.algorithm.classifier.to(self.device)

        feature_extractor.eval()
        classifier.eval()

        total_loss, preds_list, labels_list = [], [], []

        with torch.no_grad():
            for data, labels in test_loader:
                data = data.float().to(self.device)
                labels = labels.view((-1)).long().to(self.device)

                # forward pass
                if self.dataset_configs.backbone =="tf_encoder": features, _ = feature_extractor(data)
                else: features = feature_extractor(data)
                predictions = classifier(features)

                # compute loss
                loss = F.cross_entropy(predictions, labels)
                total_loss.append(loss.item())
                pred = predictions.detach()  # .argmax(dim=1)  # get the index of the max log-probability

                # append predictions and labels
                preds_list.append(pred)
                labels_list.append(labels)

        self.loss = torch.tensor(total_loss).mean()  # average loss
        self.full_preds = torch.cat((preds_list))
        self.full_labels = torch.cat((labels_list))
    

    def get_configs(self):
        dataset_class = get_dataset_class(self.dataset)
        hparams_class = get_hparams_class(self.dataset)
        return dataset_class(), hparams_class()

    def load_data(self, src_id, trg_id):
        ## Init Label Distribution (original_label distribution)
        self.src_train_dl = data_generator(self.data_path, src_id, self.dataset_configs, self.hparams, "train")
        self.src_test_dl = data_generator(self.data_path, src_id, self.dataset_configs, self.hparams, "test")
        self.trg_train_dl = data_generator(self.data_path, trg_id, self.dataset_configs, self.hparams, "train")

        # batch_size=1
        batch_size = self.hparams["batch_size"]
        self.hparams["batch_size"] = 1
        self.trg_test_dl = data_generator(self.data_path, trg_id, self.dataset_configs, self.hparams, "test")
        self.hparams["batch_size"] = batch_size

        # No drop_last & No shuffle 
        drop_last = self.dataset_configs.drop_last
        shuffle = self.dataset_configs.shuffle
        self.dataset_configs.drop_last = False
        self.hparams["batch_size"] = 1 # 
        self.trg_train_b1_dl = data_generator(self.data_path, trg_id, self.dataset_configs, self.hparams, "train")
        self.trg_test_b1_dl = data_generator(self.data_path, trg_id, self.dataset_configs, self.hparams, "test")

        self.dataset_configs.drop_last = drop_last
        self.dataset_configs.shuffle = shuffle
        self.hparams["batch_size"] = batch_size # 
        
        ## Balanced Label Distribution
        # source is okay
        self.src_balanced_train_dl = data_generator_balanced(self.data_path, src_id, self.dataset_configs, self.hparams, "train","same")
        self.src_balanced_test_dl = data_generator_balanced(self.data_path, src_id, self.dataset_configs, self.hparams, "test","same")


        ## Up-sampled balanced label distiribution
        # source is okay
        self.src_up_balanced_train_dl = data_generator_balanced(self.data_path, src_id, self.dataset_configs, self.hparams, "train","up-sampling")
        self.src_up_balanced_test_dl = data_generator_balanced(self.data_path, src_id, self.dataset_configs, self.hparams, "test","up-sampling")

        # Few-shot
        self.few_shot_dl_5 = few_shot_data_generator(self.trg_test_dl, self.dataset_configs, 5)  # set 5 to other value if you want other k-shot FST


    def create_save_dir(self, save_dir):
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

    def calculate_metrics_risks(self, label_weight=None, cluster_algorithm=None):
        # calculation based source test data
        self.evaluate('src',self.src_test_dl)
        src_risk = self.loss.item()
        # calculation based few_shot test data
        self.evaluate('few',self.few_shot_dl_5)
        fst_risk = self.loss.item()
        # calculation based target test data
        self.evaluate('trg',self.trg_test_dl, label_weight,cluster_algorithm)
        trg_risk = self.loss.item()

        # calculate metrics
        acc = self.ACC(self.full_preds.argmax(dim=1).cpu(), self.full_labels.cpu()).item()
        # f1_torch
        f1 = self.F1(self.full_preds.argmax(dim=1).cpu(), self.full_labels.cpu()).item()
        auroc = self.AUROC(self.full_preds.cpu(), self.full_labels.cpu()).item()
        # f1_sk learn
        # f1 = f1_score(self.full_preds.argmax(dim=1).cpu().numpy(), self.full_labels.cpu().numpy(), average='macro')

        risks = src_risk, fst_risk, trg_risk
        metrics = acc, f1, auroc

        return risks, metrics

    def save_tables_to_file(self,table_results, name):
        # save to file if needed
        table_results.to_csv(os.path.join(self.exp_log_dir,f"{name}.csv"))

    def save_checkpoint(self, home_path, log_dir, last_model, best_model):
        save_dict = {
            "last": last_model,
            "best": best_model
        }
        # save classification report

        # another repo
        #self.exp_log_dir = os.path.join(self.home_path, self.save_dir, self.experiment_description, f"{self.run_description}")
        #os.makedirs(self.exp_log_dir, exist_ok=True)
        
        save_path = os.path.join(home_path, log_dir, f"checkpoint.pt")
        torch.save(save_dict, save_path) # too much memory    

    def calculate_avg_std_wandb_table(self, results):

        avg_metrics = [np.mean(results.get_column(metric)) for metric in results.columns[2:]]
        std_metrics = [np.std(results.get_column(metric)) for metric in results.columns[2:]]
        summary_metrics = {metric: np.mean(results.get_column(metric)) for metric in results.columns[2:]}

        results.add_data('mean', '-', *avg_metrics)
        results.add_data('std', '-', *std_metrics)

        return results, summary_metrics

    def log_summary_metrics_wandb(self, results, risks):
       
        # Calculate average and standard deviation for metrics
        avg_metrics = [np.mean(results.get_column(metric)) for metric in results.columns[2:]]
        std_metrics = [np.std(results.get_column(metric)) for metric in results.columns[2:]]

        avg_risks = [np.mean(risks.get_column(risk)) for risk in risks.columns[2:]]
        std_risks = [np.std(risks.get_column(risk)) for risk in risks.columns[2:]]

        # Estimate summary metrics
        summary_metrics = {metric: np.mean(results.get_column(metric)) for metric in results.columns[2:]}
        summary_risks = {risk: np.mean(risks.get_column(risk)) for risk in risks.columns[2:]}


        # append avg and std values to metrics
        results.add_data('mean', '-', *avg_metrics)
        results.add_data('std', '-', *std_metrics)

        # append avg and std values to risks 
        results.add_data('mean', '-', *avg_risks)
        risks.add_data('std', '-', *std_risks)

    def wandb_logging(self, total_results, total_risks, summary_metrics, summary_risks):
        # log wandb
        wandb.log({'results': total_results})
        wandb.log({'risks': total_risks})
        wandb.log({'hparams': wandb.Table(dataframe=pd.DataFrame(dict(self.hparams).items(), columns=['parameter', 'value']), allow_mixed_types=True)})
        wandb.log(summary_metrics)
        wandb.log(summary_risks)

    def calculate_metrics(self, label_weight=None, cluster_algorithm=None):
       
        self.evaluate('trg',self.trg_test_dl,label_weight, cluster_algorithm)
        # accuracy  
        acc = self.ACC(self.full_preds.argmax(dim=1).cpu(), self.full_labels.cpu()).item()
        # f1
        f1 = self.F1(self.full_preds.argmax(dim=1).cpu(), self.full_labels.cpu()).item()
        # auroc 
        auroc = self.AUROC(self.full_preds.cpu(), self.full_labels.cpu()).item()

        return acc, f1, auroc

    def calculate_risks(self, label_weight=None, cluster_algorithm=None):
         # calculation based source test data
        self.evaluate('src',self.src_test_dl)
        src_risk = self.loss.item()
        # calculation based few_shot test data
        self.evaluate('few',self.few_shot_dl_5)
        fst_risk = self.loss.item()
        # calculation based target test data
        self.evaluate('trg',self.trg_test_dl, label_weight, cluster_algorithm)
        trg_risk = self.loss.item()

        return src_risk, fst_risk, trg_risk

    def append_results_to_tables(self, table, scenario, run_id, metrics):

        # Create metrics and risks rows
        results_row = [scenario, run_id, *metrics]

        # Create new dataframes for each row
        results_df = pd.DataFrame([results_row], columns=table.columns)

        # Concatenate new dataframes with original dataframes
        table = pd.concat([table, results_df], ignore_index=True)

        return table
    
    def add_mean_std_table(self, table, columns):
        # Calculate average and standard deviation for metrics
        avg_metrics = [table[metric].mean() for metric in columns[2:]]
        std_metrics = [table[metric].std() for metric in columns[2:]]

        # Create dataframes for mean and std values
        mean_metrics_df = pd.DataFrame([['mean', '-', *avg_metrics]], columns=columns)
        std_metrics_df = pd.DataFrame([['std', '-', *std_metrics]], columns=columns)

        # Concatenate original dataframes with mean and std dataframes
        table = pd.concat([table, mean_metrics_df, std_metrics_df], ignore_index=True)

        # Create a formatting function to format each element in the tables
        format_func = lambda x: f"{x:.4f}" if isinstance(x, float) else x

        # Apply the formatting function to each element in the tables
        table = table.applymap(format_func)

        return table 

    def visualize(self):
        feature_extractor = self.algorithm.feature_extractor.to(self.device)
        classifier = self.algorithm.classifier.to(self.device)
        feature_extractor.eval()
        
        self.src_true_labels = np.array([])
        self.src_pred_labels = np.array([])
        self.src_all_features = []
        
        self.trg_true_labels = np.array([]) 
        self.trg_pred_labels = np.array([])
        self.trg_all_features = [] 
        
        self.trg_test_true_labels = np.array([]) 
        self.trg_test_pred_labels = np.array([])
        self.trg_test_all_features = [] 
        
        with torch.no_grad():
            # source (train)
            for data, labels in self.src_train_dl:
                data = data.float().to(self.device)
                labels = labels.view((-1)).long().to(self.device)
                features = feature_extractor(data)
                pred_labels = classifier(features) 
                self.src_all_features.append(features.cpu().numpy())
                self.src_true_labels = np.append(self.src_true_labels, labels.data.cpu().numpy())
                pred_labels = pred_labels.argmax(dim=1)
                self.src_pred_labels = np.append(self.src_pred_labels, pred_labels.data.cpu().numpy())
                

            # target (train)
            for data, labels in self.trg_train_dl:
                data = data.float().to(self.device)
                labels = labels.view((-1)).long().to(self.device)
                # forward pass
                features = feature_extractor(data)
                pred_labels = classifier(features)                
                self.trg_all_features.append(features.cpu().numpy())
                self.trg_true_labels = np.append(self.trg_true_labels, labels.data.cpu().numpy())
                pred_labels = pred_labels.argmax(dim=1)
                self.trg_pred_labels = np.append(self.trg_pred_labels, pred_labels.data.cpu().numpy())

            # target (test)
            for data, labels in self.trg_test_dl:
                data = data.float().to(self.device) 
                labels = labels.view((-1)).long().to(self.device)
                features = feature_extractor(data)
                pred_labels = classifier(features)
                self.trg_test_all_features.append(features.cpu().numpy())
                self.trg_test_true_labels = np.append(self.trg_test_true_labels, labels.data.cpu().numpy())
                pred_labels = pred_labels.argmax(dim=1)
                self.trg_test_pred_labels = np.append(self.trg_test_pred_labels, pred_labels.data.cpu().numpy())
            
            self.src_all_features = np.vstack(self.src_all_features)
            self.trg_all_features = np.vstack(self.trg_all_features)
            self.trg_test_all_features = np.vstack(self.trg_test_all_features)
