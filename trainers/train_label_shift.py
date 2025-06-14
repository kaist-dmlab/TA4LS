import sys

import torch
import torch.nn.functional as F
import os
import wandb
import pandas as pd
import numpy as np
import warnings
import sklearn.exceptions
import collections
import argparse
import warnings
import sklearn.exceptions

from utils import fix_randomness, starting_logs, AverageMeter
from algorithms.algorithms import get_algorithm_class
from models.models import get_backbone_class
from trainers.abstract_trainer import AbstractTrainer
#warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)
parser = argparse.ArgumentParser()

import pickle
import copy


class Trainer_LS(AbstractTrainer):

    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.results_columns = ["scenario", "run", "acc", "f1_score", "auroc"]
        self.risks_columns = ["scenario", "run", "src_risk", "few_shot_risk", "trg_risk"]


    def fit(self):

        # table with metrics
        table_results = pd.DataFrame(columns=self.results_columns)

        # table with risks
        table_risks = pd.DataFrame(columns=self.risks_columns)


        # Trainer
        for src_id, trg_id in self.dataset_configs.scenarios:
            for run_id in range(self.num_runs):
                # fixing random seed
                fix_randomness(run_id)

                # Logging
                self.logger, self.scenario_log_dir = starting_logs(self.dataset, self.da_method, self.exp_log_dir,
                                                                src_id, trg_id, run_id)
                # Average meters
                self.loss_avg_meters = collections.defaultdict(lambda: AverageMeter())

                # Load data
                self.load_data(src_id, trg_id)
                
                # initiate the domain adaptation algorithm
                self.initialize_algorithm()

                # Train the domain adaptation algorithm: update -> update_label_shift 
                self.last_model, self.best_model, self.label_weight, self.cluster_algorithm = self.algorithm.update_label_shift(self.args, self.src_train_dl, self.trg_train_dl, 
                                                                                     self.src_balanced_train_dl, self.trg_train_b1_dl, 
                                                                                     self.loss_avg_meters, self.logger,self.hparams, 
                                                                                     self.device, self.dataset_configs,warm_up_align=False,FE_balanced=False,
                                                                                     src_train_b1_dl=self.src_train_b1_dl,trg_test_b1_dl = self.trg_test_b1_dl) 

                # Save checkpoint ##########################################################################
                #self.save_checkpoint(self.home_path, self.scenario_log_dir, self.last_model, self.best_model)
                #with open(os.path.join(self.scenario_log_dir,"label_weight.pkl"), "wb") as file:
                #    pickle.dump(self.label_weight, file)
                ############################################################################################

                # Calculate risks and metrics
                trg_test_b1_dl = copy.deepcopy(self.trg_test_b1_dl)
                logger = copy.deepcopy(self.logger)
                metrics = self.calculate_metrics(self.label_weight,self.cluster_algorithm, trg_test_b1_dl = trg_test_b1_dl, logger= logger)
                risks = self.calculate_risks(self.label_weight,self.cluster_algorithm, trg_test_b1_dl = trg_test_b1_dl, logger= logger)

                self.logger.debug(f'calculate metrics: \n {metrics}')
                self.logger.debug(f'calculate risks: \n {risks}')
                
                # Append results to tables
                scenario = f"{src_id}_to_{trg_id}"
                table_results = self.append_results_to_tables(table_results, scenario, run_id, metrics)
                table_risks = self.append_results_to_tables(table_risks, scenario, run_id, risks)

        # Calculate and append mean and std to tables
        table_results = self.add_mean_std_table(table_results, self.results_columns)
        table_risks = self.add_mean_std_table(table_risks, self.risks_columns)


        # Save tables to file if needed
        self.save_tables_to_file(table_results, 'results')
        self.save_tables_to_file(table_risks, 'risks')

    def test(self,visualization=False):
        # Results dataframes
        last_results = pd.DataFrame(columns=self.results_columns)
        best_results = pd.DataFrame(columns=self.results_columns)

        # Cross-domain scenarios
        for src_id, trg_id in self.dataset_configs.scenarios:
            for run_id in range(self.num_runs):
                # fixing random seed
                fix_randomness(run_id)

                # Logging
                self.scenario_log_dir = os.path.join(self.exp_log_dir, src_id + "_to_" + trg_id + "_run_" + str(run_id))

                self.loss_avg_meters = collections.defaultdict(lambda: AverageMeter())

                # Load data
                self.load_data(src_id, trg_id)

                # Build model
                self.initialize_algorithm()

                # Load chechpoint 
                last_chk, best_chk = self.load_checkpoint(self.scenario_log_dir)
                self.label_weight = pickle.load(os.path.join(self.scenario_log_dir,"label_weight.pkl"))
                self.cluster_algorithm = pickle.load(os.path.join(self.scenario_log_dir,"cluster_algorithm.pkl"))

                # Testing the last model
                self.algorithm.network.load_state_dict(last_chk)
                self.evaluate('trg',self.trg_test_dl, self.label_weight) 
                last_metrics = self.calculate_metrics(self.label_weight)
                last_results = self.append_results_to_tables(last_results, f"{src_id}_to_{trg_id}", run_id,
                                                             last_metrics)
                
                # Testing the best model
                self.algorithm.network.load_state_dict(best_chk)
                self.evaluate('trg',self.trg_test_dl, self.label_weight,self.cluster_algorithm)
                best_metrics = self.calculate_metrics(self.label_weight,self.cluster_algorithm)
                # Append results to tables
                best_results = self.append_results_to_tables(best_results, f"{src_id}_to_{trg_id}", run_id,best_metrics) 


        last_scenario_mean_std = last_results.groupby('scenario')[['acc', 'f1_score', 'auroc']].agg(['mean', 'std'])
        best_scenario_mean_std = best_results.groupby('scenario')[['acc', 'f1_score', 'auroc']].agg(['mean', 'std'])


        # Save tables to file if needed
        self.save_tables_to_file(last_scenario_mean_std, 'last_results')
        self.save_tables_to_file(best_scenario_mean_std, 'best_results')

        # printing summary 
        summary_last = {metric: np.mean(last_results[metric]) for metric in self.results_columns[2:]}
        summary_best = {metric: np.mean(best_results[metric]) for metric in self.results_columns[2:]}
        for summary_name, summary in [('Last', summary_last), ('Best', summary_best)]:
            for key, val in summary.items():
                print(f'{summary_name}: {key}\t: {val:2.4f}')


