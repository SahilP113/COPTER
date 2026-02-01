
import pandas as pd
import numpy as np
import os, wandb, random

from setup import create_parser, get_hparams, setup_paths
from data_prep.read_data import load_data
from train_utils_ET import training_and_validation

from metrics.metrics_utils_TCR import get_metrics, get_total_metrics, save_torch_train_val_preds, save_results
#from data_prep import process_and_split_data

import torch
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils import shuffle
#import data_prep
import pickle
import h5py

os.environ['OPENBLAS_NUM_THREADS'] = '1'

FOLDSEEK_MISSING_IDX = 20



#def run_finetune(embed, celltype_dict, celltype_protein_dict, positive_proteins, negative_proteins, data_split_path, random_state, models_output_dir, embed_name, embed_name2, hparams, batch_size, num_epoch, train_size, val_size, weigh_sample, weigh_loss):
def run_finetune(models_output_dir, plm_type, embed_name, embed_name2, hparams, batch_size, num_epoch, train_size, val_size, weigh_sample, weigh_loss):

    # Training and validation
    overall_stats_list = []
    positive_proportion_train_list = []
    positive_proportion_test_list = []
    accuracy_list = []
    f1_score_list = []
    apr_5_list = []
    apr_10_list = []
    apr_15_list = []
    apr_20_list = []
    auroc_scores_list = []
    ap_scores_list = []
    bal_acc_list = []
    test_recall_5_list = [] 
    test_precision_5_list = []
    test_ap_5_list = [] 
    test_recall_10_list = [] 
    test_precision_10_list = []
    test_ap_10_list = []

    overall_stats = {}
    positive_proportion_train = {}
    accuracy = {}
    f1_score = {}
    apr_5 = {}
    apr_10 = {}
    apr_15 = {}
    apr_20 = {}
    auroc_scores = {}
    ap_scores = {}
    bal_acc = {}
    positive_proportion_test = {}
    test_recall_5 = {}
    test_precision_5 = {}
    test_ap_5 = {}
    test_recall_10 = {}
    test_precision_10 = {}
    test_ap_10 = {}

    
    for i in range(5):

        splits_folder = "pan_prep_data/PanPrepSplits/"

        train_path = splits_folder + "train" + str(i) + ".csv"
        val_path = splits_folder + "val" + str(i) + ".csv"
        test_path = splits_folder + "test" + str(i) + ".csv"

        train_data = pd.read_csv(train_path)
        val_data = pd.read_csv(val_path)
        test_data = pd.read_csv(test_path)

        TCR_train = list(train_data["binding_TCR"])
        epitope_train = list(train_data["peptide"])
        label_train = torch.tensor(train_data["label"])

        TCR_val = list(val_data["binding_TCR"])
        epitope_val = list(val_data["peptide"])
        label_val = torch.tensor(val_data["label"])

        TCR_test = dict()
        epitopes_test = dict()
        labels_test = dict()

        TCR_test2 = list(test_data["binding_TCR"])
        epitopes_test2 = list(test_data["peptide"])
        labels_test2 = torch.tensor(test_data["label"])

        unique_epitopes = np.unique(test_data["peptide"])

        for epitope in unique_epitopes:
            TCR_list = []
            epitopes_list = []
            labels_list = []
            
            for row in range(test_data.shape[0]):
                if test_data.iloc[row]["peptide"] == epitope:
                    TCR = test_data.iloc[row]["binding_TCR"]
                    label = test_data.iloc[row]["label"]
                    TCR_list.append(TCR)
                    epitopes_list.append(epitope)
                    labels_list.append(label)
            if len(TCR_list) > 0:
                TCR_test[epitope] = TCR_list
                epitopes_test[epitope] = epitopes_list
                labels_test[epitope] = np.array(labels_list)

        clf, pooling_clf, best_epoch, best_val_auprc = finetune_train_stage(TCR_train, epitope_train, label_train, TCR_val, epitope_val, label_val, plm_type, hparams, num_epoch, batch_size, weigh_sample, weigh_loss, models_output_dir, embed_name)


        positive_proportion_train = {}
        positive_proportion_train['celltype'] = sum(label_train) / len(label_train)
        wandb.log({f'train positive proportion celltype': positive_proportion_train['celltype'], 'best_val_auprc': best_val_auprc})
        overall_stats, positive_proportion_test, accuracy, f1_score, apr_5, apr_10, apr_15, apr_20, auroc_scores, ap_scores, bal_acc, test_recall_5, test_precision_5, test_ap_5, test_recall_10, test_precision_10, test_ap_10 = finetune_evaluate(unique_epitopes, clf, pooling_clf, TCR_test, epitopes_test, labels_test, TCR_test2, epitopes_test2, labels_test2, plm_type, models_output_dir, embed_name)

        overall_stats_list.append(overall_stats)
        positive_proportion_train_list.append(positive_proportion_train)
        positive_proportion_test_list.append(positive_proportion_test)
        accuracy_list.append(accuracy)
        f1_score_list.append(f1_score)
        apr_5_list.append(apr_5)
        apr_10_list.append(apr_10)
        apr_15_list.append(apr_15)
        apr_20_list.append(apr_20)
        auroc_scores_list.append(auroc_scores)
        ap_scores_list.append(ap_scores)
        bal_acc_list.append(bal_acc)
        test_recall_5_list.append(test_recall_5)
        test_precision_5_list.append(test_precision_5)
        test_ap_5_list.append(test_ap_5)
        test_recall_10_list.append(test_recall_10)
        test_precision_10_list.append(test_precision_10)
        test_ap_10_list.append(test_ap_10)


        save_path = os.path.join(models_output_dir, f"{embed_name}_model" + str(i) + ".pt")
        save_path2 = os.path.join(models_output_dir, f"{embed_name2}_model" + str(i) + ".pt")

        #torch.save({'epoch': best_epoch, 'model_state_dict': clf.state_dict(), 'pooling_state_dict': pooling_clf.state_dict()}, save_path)
        torch.save({'epoch': best_epoch, 'model_state_dict': clf.state_dict()}, save_path)
        torch.save({'epoch': best_epoch, 'pooling_state_dict': pooling_clf.state_dict()}, save_path2)

    mean_stats = get_combined_dict(overall_stats_list)
    std_dev_stats = get_sd_dict(overall_stats_list)

    positive_proportion_train = get_combined_dict(positive_proportion_train_list) 
    positive_proportion_test = get_combined_dict(positive_proportion_test_list)
    accuracy = get_combined_dict(accuracy_list)
    f1_score = get_combined_dict(f1_score_list)
    apr_5 = get_combined_dict(apr_5_list)
    apr_10 = get_combined_dict(apr_10_list)
    apr_15 = get_combined_dict(apr_15_list)
    apr_20 = get_combined_dict(apr_20_list)
    auroc_scores = get_combined_dict(auroc_scores_list)
    ap_scores = get_combined_dict(ap_scores_list)
    bal_acc = get_combined_dict(bal_acc_list)
    test_recall_5 = get_combined_dict(test_recall_5_list)
    test_precision_5 = get_combined_dict(test_precision_5_list)
    test_ap_5 = get_combined_dict(test_ap_5_list)
    test_recall_10 = get_combined_dict(test_recall_10_list)
    test_precision_10 = get_combined_dict(test_precision_10_list)
    test_ap_10 = get_combined_dict(test_ap_10_list)

    return mean_stats, std_dev_stats, positive_proportion_train, positive_proportion_test, accuracy, f1_score, apr_5, apr_10, apr_15, apr_20, auroc_scores, ap_scores, bal_acc, test_recall_5, test_precision_5, test_ap_5, test_recall_10, test_precision_10, test_ap_10


def get_combined_dict(dict_list):
    temp_dict = {}
    for dictionary in dict_list:
        for epitope, metric in dictionary.items():
            if epitope not in temp_dict:
                temp_dict[epitope] = []
            temp_dict[epitope].append(metric)
    
    combined_dict = {epitope : np.mean(metrics) for epitope, metrics in temp_dict.items()} 
    return combined_dict

def get_sd_dict(dict_list):
    temp_dict = {}
    for dictionary in dict_list:
        for epitope, metric in dictionary.items():
            if epitope not in temp_dict:
                temp_dict[epitope] = []
            temp_dict[epitope].append(metric)

    std_dev_dict = {epitope : np.std(metrics, ddof=1) for epitope, metrics in temp_dict.items()}
    return std_dev_dict


#def finetune_train_stage(X_train, context_train, y_train, random_state, groups_train, cts_train, hparams, train_size, val_size, num_epoch, batch_size, weigh_sample, weigh_loss, models_output_dir, embed_name):
def finetune_train_stage(TCR_train, epitope_train, label_train, TCR_val, epitope_val, label_val, plm_type, hparams, num_epoch, batch_size, weigh_sample, weigh_loss, models_output_dir, embed_name):

    clf, pooling_clf, best_train_y, best_train_preds, best_val_y, best_val_preds, best_epoch, best_val_auprc = training_and_validation(TCR_train, TCR_val, epitope_train, epitope_val, label_train, label_val, plm_type, num_epoch, batch_size, weigh_sample, weigh_loss, hparams)

    clf = clf.to(torch.device('cpu'))
    pooling_clf = pooling_clf.to(torch.device('cpu'))
    return clf, pooling_clf, best_epoch, best_val_auprc



def get_padded_tcr(tcr_batch, batch_max, plm_type):

    data_folder = "data/pan_prep_data"

    if plm_type == "esm2_8m":
        embeddings_file = "ESM_esm2_t6_8M_UR50D_features_TCR.h5"
    elif plm_type == "esm2_650m":
        embeddings_file = "ESM_esm2_t33_650M_UR50D_features_TCR.h5"
    elif plm_type == "progen2_small":
        embeddings_file = "ProGen_progen2-small_features_TCR.h5"
    elif plm_type == "progen2_large":
        embeddings_file = "ProGen_progen2-large_features_TCR.h5"
    else:
        raise ValueError(f"Unknown plm_type: {plm_type}")
    
    tcr_tokenized_file = os.path.join(data_folder, embeddings_file)
    tcr_h5file = h5py.File(tcr_tokenized_file, 'r')


    tcr_embeddings = []
    for tcr_sequence in tcr_batch:
        if tcr_sequence not in tcr_h5file.keys():
            print("Missing Sequence: ", tcr_sequence)
            continue
        tcr_h5dataset = tcr_h5file[tcr_sequence]  
        esm_embed = torch.tensor(np.array(tcr_h5dataset))
        padded_embed = torch.nn.functional.pad(esm_embed, (0, 0, 0, batch_max - esm_embed.shape[0]), value=FOLDSEEK_MISSING_IDX)
        tcr_embeddings.append(padded_embed)
    tcr_embed_tensor = torch.stack(tcr_embeddings)
    return tcr_embed_tensor

def get_max_length(tcr_batch):
    batch_max_length = 0
    for tcr_sequence in tcr_batch:
        sequence_length = len(tcr_sequence)
        if sequence_length > batch_max_length:
            batch_max_length = sequence_length
    return batch_max_length

def get_epitope(epitope_batch, plm_type):

    data_folder = "data/pan_prep_data"

    if plm_type == "esm2_8m":
        embeddings_file = "ESM_esm2_t6_8M_UR50D_features_epitope.h5"
    elif plm_type == "esm2_650m":
        embeddings_file = "ESM_esm2_t33_650M_UR50D_features_epitope.h5"
    elif plm_type == "progen2_small":
        embeddings_file = "ProGen_progen2-small_features_epitope.h5"
    elif plm_type == "progen2_large":
        embeddings_file = "ProGen_progen2-large_features_epitope.h5"
    else:
        raise ValueError(f"Unknown plm_type: {plm_type}")

    epitope_tokenized_file = os.path.join(data_folder, embeddings_file)
    epitope_h5file = h5py.File(epitope_tokenized_file, 'r')

    epitope_embeddings = []
    for epitope in epitope_batch:
        epitope_h5dataset = epitope_h5file[epitope]  
        esm_embed = torch.tensor(np.array(epitope_h5dataset))
        avg_embed = torch.mean(esm_embed, dim = 0)
        epitope_embeddings.append(avg_embed)
    epitope_tensor = torch.stack(epitope_embeddings)
    return epitope_tensor



#def finetune_evaluate(celltype_protein_dict, clf, pooling_clf, X_test, context_test, y_test, groups_test, models_output_dir, embed_name, train_ranks, val_ranks):
def finetune_evaluate(unique_epitopes, clf, pooling_clf, TCR_test, epitope_test, label_test, TCR_test2, epitope_test2, label_test2, plm_type, models_output_dir, embed_name):

    accuracy = {}
    f1_score = {}
    apr_5 = {}
    apr_10 = {}
    apr_15 = {}
    apr_20 = {}
    auroc_scores = {}
    ap_scores = {}
    bal_acc = {}
    positive_proportion_test = {}
    test_recall_5 = {}
    test_precision_5 = {}
    test_ap_5 = {}
    test_recall_10 = {}
    test_precision_10 = {}
    test_ap_10 = {}

    test_ranks = {}
    overall_stats = {}



    clf.eval()
    with torch.no_grad():
        test_max = get_max_length(TCR_test2)                   #If the initial data contains the gene names
        new_tcr_test = get_padded_tcr(TCR_test2, test_max, plm_type)  #If the initial data contains the gene names
        mask = (new_tcr_test != FOLDSEEK_MISSING_IDX)[:, :, 0]
        epitope = get_epitope(epitope_test2, plm_type)
        pooled_tcr_test = pooling_clf(new_tcr_test, epitope, mask)
        y_pred_total = torch.sigmoid(clf(pooled_tcr_test)).squeeze(-1).numpy()

    accuracy_total, f1_score_total, auroc_total, auprc_total = get_total_metrics(label_test2, y_pred_total)
    overall_stats['total accuracy'] = accuracy_total
    overall_stats['total f1-score'] = f1_score_total
    overall_stats['total AUROC'] = auroc_total
    overall_stats['total AUPRC'] = auprc_total


    for epitope_type in unique_epitopes:
    #for celltype in celltype_protein_dict:
        if epitope_type not in TCR_test: continue
        clf.eval()
        with torch.no_grad():
            test_max = get_max_length(TCR_test[epitope_type])                   #If the initial data contains the gene names
            new_tcr_test = get_padded_tcr(TCR_test[epitope_type], test_max, plm_type)  #If the initial data contains the gene names
            mask = (new_tcr_test != FOLDSEEK_MISSING_IDX)[:, :, 0]
            epitope = get_epitope(epitope_test[epitope_type], plm_type)
            pooled_tcr_test = pooling_clf(new_tcr_test, epitope, mask)
            y_test_pred = torch.sigmoid(clf(pooled_tcr_test)).squeeze(-1).numpy()
        
        # Evaluation on test set
        accuracy[epitope_type], f1_score[epitope_type], apr_5[epitope_type], apr_10[epitope_type], apr_15[epitope_type], apr_20[epitope_type], auroc_scores[epitope_type], ap_scores[epitope_type], bal_acc[epitope_type], test_recall_5[epitope_type], test_precision_5[epitope_type], test_ap_5[epitope_type], test_recall_10[epitope_type], test_precision_10[epitope_type], test_ap_10[epitope_type], sorted_y_test, sorted_preds_test, sorted_groups_test, positive_proportion_test[epitope_type] = get_metrics(label_test, y_test_pred, epitope_type)
        #apr_5[celltype], apr_10[celltype], apr_15[celltype], apr_20[celltype], auroc_scores[celltype], ap_scores[celltype], test_recall_5[celltype], test_precision_5[celltype], test_ap_5[celltype], test_recall_10[celltype], test_precision_10[celltype], test_ap_10[celltype], sorted_y_test, sorted_preds_test, sorted_groups_test, positive_proportion_test[celltype] = get_metrics(y_test, y_test_pred, groups_test, celltype)
        wandb.log({f'test APR@5 {epitope_type}': apr_5[epitope_type],
                   f'test APR@10 {epitope_type}': apr_10[epitope_type], 
                   f'test APR@15 {epitope_type}': apr_15[epitope_type], 
                   f'test APR@20 {epitope_type}': apr_20[epitope_type],  
                   f'test AUPRC {epitope_type}': ap_scores[epitope_type], 
                   f'test AUROC {epitope_type}': auroc_scores[epitope_type],
                   f'test accuracy {epitope_type}': accuracy[epitope_type],
                   f'test F1-score {epitope_type}': f1_score[epitope_type],
                   f'test balanced accuracy {epitope_type}': bal_acc[epitope_type],
                   f'test positive proportion {epitope_type}': positive_proportion_test[epitope_type],
                   f'test positive number {epitope_type}': sum(label_test[epitope_type]),
                   f'test total number {epitope_type}': len(label_test[epitope_type]),
                   f'test recall@5 {epitope_type}': test_recall_5[epitope_type],
                   f'test precision@5 {epitope_type}': test_precision_5[epitope_type],
                   f'test AP@5 {epitope_type}': test_ap_5[epitope_type],
                   f'test recall@10 {epitope_type}': test_recall_10[epitope_type],
                   f'test precision@10 {epitope_type}': test_precision_10[epitope_type],
                   f'test AP@10 {epitope_type}': test_ap_10[epitope_type]})
    
    return overall_stats, positive_proportion_test, accuracy, f1_score, apr_5, apr_10, apr_15, apr_20, auroc_scores, ap_scores, bal_acc, test_recall_5, test_precision_5, test_ap_5, test_recall_10, test_precision_10, test_ap_10


def main(args, hparams, wandb):
    print(args)

    # Set up model environment and data/model paths
    models_output_dir, metrics_output_dir, random_state, embed_path, labels_path = setup_paths(args)
    
    # Run model
    data_split_path = args.data_split_path + ".json"
    mean_stats, std_dev_stats, positive_proportion_train, positive_proportion_test, accuracy, f1_score, apr_5, apr_10, apr_15, apr_20, auroc_scores, ap_scores, bal_acc, test_recall_5, test_precision_5, test_ap_5, test_recall_10, test_precision_10, test_ap_10 = run_finetune(models_output_dir, args.plm, args.embed, args.embed2, hparams, args.batch_size, args.num_epoch, args.train_size, args.val_size, args.weigh_sample, args.weigh_loss)
    #positive_proportion_train, positive_proportion_test, apr_5, apr_10, apr_15, apr_20, auroc_scores, ap_scores, test_recall_5, test_precision_5, test_ap_5, test_recall_10, test_precision_10, test_ap_10 = run_finetune(embed, celltype_dict, celltype_protein_dict, positive_proteins, negative_proteins, data_split_path, random_state, models_output_dir, args.embed, args.embed2, hparams, args.batch_size, args.num_epoch, args.train_size, args.val_size, args.weigh_sample, args.weigh_loss)

    # Generate outputs and plots
    print("Finished evaluation, generating plots...\n")
    output_results_path = os.path.join(metrics_output_dir, f"{args.embed}_{args.task_name}_results_ET_plarge.csv")  #Chnaged from results to results2 to results3 create a new results file
    output_figs_path = os.path.join(metrics_output_dir, f"{args.embed}_{args.task_name}_")
    save_results(output_results_path, mean_stats, std_dev_stats, accuracy, f1_score, apr_5, apr_10, apr_15, apr_20, ap_scores, auroc_scores, bal_acc, test_ap_5, test_recall_5, test_precision_5, test_ap_10, test_recall_10, test_precision_10)
    

if __name__ == '__main__':
    args = create_parser()

    if not args.random:
        np.random.seed(args.random_state)
        random.seed(args.random_state)
        torch.manual_seed(args.random_state)
        torch.cuda.manual_seed(args.random_state)
        torch.backends.cudnn.deterministic = True

    hparams_raw = get_hparams(args)
    print(hparams_raw)

    wandb.init(config = hparams_raw, project = "finetune", entity = "sahilp113-duke-university") #entity="pinnacle"

    hparams = wandb.config

    main(args, hparams, wandb)