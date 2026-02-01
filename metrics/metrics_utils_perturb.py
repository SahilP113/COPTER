from typing import Dict
import numpy as np
import pandas as pd
import torch.nn.functional as F

import json, matplotlib, os

import torch
import scanpy as sc
from sklearn.metrics import average_precision_score, roc_auc_score, balanced_accuracy_score, f1_score, accuracy_score, precision_score, recall_score
from sklearn.metrics import r2_score
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae

import matplotlib.pyplot as plt
import seaborn as sns
matplotlib.use('Agg')



"""
def save_results(output_results_path: str, mse):

    #print("In save_results")
    res = {'MSE': mse}
    with open(output_results_path, 'w') as f:
        json.dump(res, f)        
    print(f"\nResults output to -> {output_results_path}")
"""

def save_results(output_results_path: str, test_metrics):
    """
    Save results in the form of dictionary to a json file.
    """
    #print("In save_results")
    #res = {'MSE': mse}
    test_metrics_clean = {k: float(v) for k, v in test_metrics.items()}
    with open(output_results_path, 'w') as f:
        json.dump(test_metrics_clean, f)        
    print(f"\nResults output to -> {output_results_path}")



def compute_metrics(results):
    """
    Given results from a model run and the ground truth, compute metrics

    """
    metrics = {}
    metrics_pert = {}

    metric2fct = {
           'mse': mse,
           'pearson': pearsonr
    }
    
    for m in metric2fct.keys():
        metrics[m] = []
        metrics[m + '_de'] = []

    for pert in np.unique(results['pert_cat']):

        metrics_pert[pert] = {}
        p_idx = np.where(results['pert_cat'] == pert)[0]
            
        for m, fct in metric2fct.items():
            if m == 'pearson':
                val = fct(results['pred'][p_idx].mean(0), results['truth'][p_idx].mean(0))[0]
                if np.isnan(val):
                    val = 0
            else:
                val = fct(results['pred'][p_idx].mean(0), results['truth'][p_idx].mean(0))

            metrics_pert[pert][m] = val
            metrics[m].append(metrics_pert[pert][m])

       
        if pert != 'ctrl':
            
            for m, fct in metric2fct.items():
                if m == 'pearson':
                    val = fct(results['pred_de'][p_idx].mean(0), results['truth_de'][p_idx].mean(0))[0]
                    if np.isnan(val):
                        val = 0
                else:
                    val = fct(results['pred_de'][p_idx].mean(0), results['truth_de'][p_idx].mean(0))
                    
                metrics_pert[pert][m + '_de'] = val
                metrics[m + '_de'].append(metrics_pert[pert][m + '_de'])

        else:
            for m, fct in metric2fct.items():
                metrics_pert[pert][m + '_de'] = 0
    
    for m in metric2fct.keys():
        
        metrics[m] = np.mean(metrics[m])
        metrics[m + '_de'] = np.mean(metrics[m + '_de'])
    
    return metrics, metrics_pert



def calculate_mse20_old(all_pred, all_y):

    all_mse = []
    for i in range(all_y.shape[0]):
        y = all_y[i]
        preds = all_pred[i]

        y_abs = torch.abs(y)
        top20_indices = torch.topk(y_abs, 20).indices

        top20_pred = preds[top20_indices]
        top20_y = y[top20_indices]

        mse = F.mse_loss(top20_pred, top20_y)
        all_mse.append(mse)
    
    all_mse = torch.stack(all_mse)
    final_mse = all_mse.mean()
    final_mse = final_mse.item()
    return final_mse

def calculate_mse20(all_pred_deg, all_y_deg):

    all_mse = []
    for i in range(all_y_deg.shape[0]):
        y = all_y_deg[i]
        preds = all_pred_deg[i]

        #y_abs = torch.abs(y)
        #top20_indices = torch.topk(y_abs, 20).indices

        #top20_pred = preds[top20_indices]
        #top20_y = y[top20_indices]

        #mse = F.mse_loss(top20_pred, top20_y)
        mse = F.mse_loss(preds, y)
        all_mse.append(mse)
    
    all_mse = torch.stack(all_mse)
    final_mse = all_mse.mean()
    final_mse = final_mse.item()
    return final_mse



