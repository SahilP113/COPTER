from typing import Dict
import numpy as np
import pandas as pd

import json, matplotlib, os

import torch
from sklearn.metrics import average_precision_score, roc_auc_score, balanced_accuracy_score, f1_score, accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
matplotlib.use('Agg')

#from data_prep import process_and_split_data

#test_recall_5, test_precision_5, test_ap_5, test_recall_10, test_precision_10, test_ap_10

def save_results(output_results_path: str, mean_stats: Dict[str, Dict[str, float]], std_dev_stats: Dict[str, Dict[str, float]], accuracy: Dict[str, Dict[str, float]], f1: Dict[str, Dict[str, float]], apr_5: Dict[str, Dict[str, float]], apr_10: Dict[str, Dict[str, float]], apr_15: Dict[str, Dict[str, float]], apr_20: Dict[str, Dict[str, float]], ap_scores: Dict[str, Dict[str, float]], auroc_scores: Dict[str, Dict[str, float]], bal_acc: Dict[str, Dict[str, float]], ap_5: Dict[str, Dict[str, float]], recall_5: Dict[str, Dict[str, float]], precision_5: Dict[str, Dict[str, float]], ap_10: Dict[str, Dict[str, float]], recall_10: Dict[str, Dict[str, float]], precision_10: Dict[str, Dict[str, float]]):
    """
    Save results in the form of dictionary to a json file.
    """
    print("In save_results")
    res = {'mean': mean_stats, 'std dev': std_dev_stats, 'accuracy': accuracy, 'f1-score': f1, 'auroc': auroc_scores, 'ap': ap_scores, 'balanced_accuracy': bal_acc, 'apr5': apr_5, 'apr10': apr_10, 'apr15': apr_15, 'apr20': apr_20, 'ap5': ap_5, 'recall5': recall_5, 'precision5': precision_5, 'ap10': ap_10, 'recall10': recall_10, 'precision10': precision_10}
    with open(output_results_path, 'w') as f:
        json.dump(res, f)        
    print(f"\nResults output to -> {output_results_path}")


def save_torch_train_val_preds(best_train_y, best_train_preds, best_train_groups, best_train_cts, best_val_y, best_val_preds, best_val_groups, best_val_cts, groups_map_train, groups_map_val, cts_map_train, cts_map_val, models_output_dir, embed_name, wandb):
    train_ranks = {}
    val_ranks = {}
    for ct in np.unique(best_val_cts):
        hits = np.where(best_val_cts==ct)[0]
        val_y_ct = best_val_y[hits]
        val_preds_ct = best_val_preds[hits]
        val_groups_ct = best_val_groups[hits]
        ct_name = cts_map_val[ct]

        if len(np.unique(val_y_ct)) < 2:
            apr_5, apr_10, apr_15, apr_20, auroc_score, ap_score, ct_recall_5, ct_precision_5, ct_ap_5, ct_recall_10, ct_precision_10, ct_ap_10, sorted_val_y_ct, sorted_val_preds_ct, sorted_val_groups_ct, positive_proportion_val = -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, np.array([-1] * len(val_y_ct)), np.array([-1] * len(val_y_ct)), np.array([-1] * len(val_y_ct)), -1
        else:
            apr_5, apr_10, apr_15, apr_20, auroc_score, ap_score, ct_recall_5, ct_precision_5, ct_ap_5, ct_recall_10, ct_precision_10, ct_ap_10, sorted_val_y_ct, sorted_val_preds_ct, sorted_val_groups_ct, positive_proportion_val = get_metrics(val_y_ct, val_preds_ct, val_groups_ct, "training")
            if len(sorted_val_y_ct) > 0: sorted_val_y_ct = sorted_val_y_ct.squeeze(-1)
            if len(sorted_val_preds_ct) > 0: sorted_val_preds_ct = sorted_val_preds_ct.squeeze(-1)
        
        temp = pd.DataFrame({'y':sorted_val_y_ct, 'preds':sorted_val_preds_ct, 'name':[groups_map_val[prot_ind] for prot_ind in sorted_val_groups_ct]})
        temp['type'] = ['val'] * len(temp)
        val_ranks[ct_name] = temp
        temp.to_csv(f'{models_output_dir}/{embed_name}_val_preds_{ct_name}.csv', index=False)  # Save the validation predictions

        wandb.log({f'val APR@5 cell types {ct_name}': apr_5, 
                   f'val APR@10 cell types {ct_name}': apr_10, 
                   f'val APR@15 cell types {ct_name}': apr_15, 
                   f'val APR@20 cell types {ct_name}': apr_20, 
                   f'val AUPRC cell types {ct_name}': ap_score, 
                   f'val AUROC cell types {ct_name}': auroc_score,
                   f'val recall@5 cell types {ct_name}': ct_recall_5,
                   f'val precision@5 cell types {ct_name}': ct_precision_5,
                   f'val AP@5 cell types {ct_name}': ct_ap_5,
                   f'val recall@10 cell types {ct_name}': ct_recall_10, 
                   f'val precision@10 cell types {ct_name}': ct_precision_10,
                   f'val AP@10 cell types {ct_name}': ct_ap_10,
                   f'val positive proportion {ct_name}': positive_proportion_val})
            
    for ct in np.unique(best_train_cts): 
        hits = np.where(best_train_cts == ct)[0]
        train_y_ct = best_train_y[hits]
        train_preds_ct = best_train_preds[hits]
        train_groups_ct = best_train_groups[hits]
        _, _, _, (sorted_train_y_ct, sorted_train_preds_ct, sorted_train_groups_ct) = precision_recall_at_k(train_y_ct, train_preds_ct, k=10, prots=train_groups_ct)

        ct_name = cts_map_train[ct]
        temp = pd.DataFrame({'y': sorted_train_y_ct.squeeze(-1), 'preds':sorted_train_preds_ct.squeeze(-1), 'name':[groups_map_train[prot_ind] for prot_ind in sorted_train_groups_ct]})
        temp['type'] = ['train'] * len(temp)
        train_ranks[ct_name] = temp
        temp.to_csv(f'{models_output_dir}/{embed_name}_train_preds_{ct_name}.csv', index=False)  # Save the validation predictions
    
    return train_ranks, val_ranks


def precision_recall_at_k(y: np.ndarray, preds: np.ndarray, k: int = 10, prots: np.ndarray = None):
    """
    Calculate recall@k, precision@k, and AP@k for binary classification.
    """
    assert preds.shape[0] == y.shape[0]
    assert k > 0
    if k > preds.shape[0]: return -1, -1, -1, ([], [], [])
    
    # Sort the scores and the labels by the scores
    sorted_indices = np.argsort(preds.flatten())[::-1]
    sorted_preds = preds[sorted_indices]
    sorted_y = y[sorted_indices]
    if prots is not None:
        sorted_prots = prots[sorted_indices]
    else: sorted_prots = None
    
    # Get the scores of the k highest predictions
    topk_preds = sorted_preds[:k]
    topk_y = sorted_y[:k]
    
    # Calculate the recall@k and precision@k
    recall_k = np.sum(topk_y) / np.sum(y)
    precision_k = np.sum(topk_y) / k
    
    # Calculate the AP@k
    ap_k = average_precision_score(topk_y, topk_preds)

    return recall_k, precision_k, ap_k, (sorted_y, sorted_preds, sorted_prots)

def apr_at_k(y: np.ndarray, preds: np.ndarray, highest_k):

    #Include max_k as a parameter
    #Double check calculation of r
    #max_k = preds.shape[0]
    #r = np.sum(y)     
    pr_sum = 0
    r = 0
    preds_size = preds.shape[0]
    k_max = min(highest_k, preds_size)

    sorted_indices = np.argsort(preds.flatten())[::-1]
    sorted_y = y[sorted_indices]
    #When k = 20, an index out of bounds erro occurs. This means that the size of preds must be less than 20 for some celltypes. 
    
    for k in range(1, k_max+1):
        topk_y = sorted_y[:k]
        precision_k = np.sum(topk_y) / k
        pr_sum = pr_sum + (precision_k * sorted_y[k-1]) 
        r = r + sorted_y[k-1] 
    
    if r == 0:
        return 0
    else:
        apr_k = pr_sum/r
        return apr_k

def get_total_metrics(y, y_pred):
    y_pred_binary = (y_pred >= 0.5).astype(int)
    accuracy = accuracy_score(y, y_pred_binary)

    #for value in y_pred:
        #if value < 0.5:
            #print("Value: ", value)

    #true_positives = torch.sum(torch.tensor(y))
    #pred_positives = torch.sum(torch.tensor(y_pred_binary))
    #print("True Positives: ", true_positives)
    #print("Predicted Positives: ", pred_positives)

    #precision = precision_score(y, y_pred_binary)
    #recall = recall_score(y, y_pred_binary)

    #print("Precision: ", precision)
    #print("Recall: ", recall)

    f1 = f1_score(y, y_pred_binary)
    #print("Sklearn F1: ", f1)

    #f1_calc = (2*precision*recall)/(precision+recall)
    #print("Calculated F1: ", f1_calc)

    #f1 = f1_score(y, y_pred)
    auroc_score = roc_auc_score(y, y_pred) 
    ap_score = average_precision_score(y, y_pred)
    return accuracy, f1, auroc_score, ap_score



def get_metrics(y, y_pred, celltype):
#def get_metrics(y, y_pred, groups, celltype):
    if celltype in ["training"]:
        y = {celltype: y}
        #groups = {celltype: groups}
    
    if len(np.unique(y[celltype])) < 2:
        #print("Epitope: ", celltype)
        #print("List: ", y[celltype])
        auroc_score = float('nan')
    else:
        auroc_score = roc_auc_score(y[celltype], y_pred) 
    ap_score = average_precision_score(y[celltype], y_pred)

    #Thresholding the predictions for the balanced accuracy score. >= 0.5 is classified as 1, < 0.5 is classified as 0
    y_pred_binary = (y_pred >= 0.5).astype(int)
    accuracy = accuracy_score(y[celltype], y_pred_binary)
    f1 = f1_score(y[celltype], y_pred_binary)

    bal_acc = balanced_accuracy_score(y[celltype], y_pred_binary)
    
    recall_5, precision_5, ap_5, _ = precision_recall_at_k(y[celltype], y_pred, k=5)
    apr_5 = apr_at_k(y[celltype], y_pred, highest_k=5)
    apr_10 = apr_at_k(y[celltype], y_pred, highest_k=10)
    apr_15 = apr_at_k(y[celltype], y_pred, highest_k=15)
    apr_20 = apr_at_k(y[celltype], y_pred, highest_k=20)
    recall_10, precision_10, ap_10, (sorted_y, sorted_preds, sorted_groups) = precision_recall_at_k(y[celltype], y_pred, k=10)
    #recall_10, precision_10, ap_10, (sorted_y, sorted_preds, sorted_groups) = precision_recall_at_k(y[celltype], y_pred, k=10, prots=np.array(groups[celltype]))

    # Calculate positive label proportions for each cell type, i.e. baseline for AP metric
    positive_proportion = sum(y[celltype]) / len(y[celltype])

    return accuracy, f1, apr_5, apr_10, apr_15, apr_20, auroc_score, ap_score, bal_acc, recall_5, precision_5, ap_5, recall_10, precision_10, ap_10, sorted_y, sorted_preds, sorted_groups, positive_proportion 
    #return apr_5, apr_10, apr_15, apr_20, auroc_score, ap_score, recall_5, precision_5, ap_5, recall_10, precision_10, ap_10, sorted_y, sorted_preds, sorted_groups, positive_proportion