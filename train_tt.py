
import pandas as pd
import numpy as np
import os, wandb, random
import json
import anndata as ad

from setup import create_parser, get_hparams, setup_paths
from data_prep.read_data import load_data
from train_utils_tt import training_and_validation
from metrics.metrics_utils_tt import get_metrics, get_context_specific_metrics, get_context_free_metrics, save_torch_train_val_preds, save_results
from data_prep.data_prep_tt import process_and_split_data
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import Dataset


#import os
#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

import torch
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils import shuffle
import data_prep
import pickle
import h5py

class CustomDataset(Dataset):
    def __init__(self, *data_tuple):
        self.data_tuple = data_tuple
    
    def __len__(self):
        return len(self.data_tuple[0])

    def __getitem__(self, idx):
        X = self.data_tuple[0][idx]
        return tuple(data[idx] for data in self.data_tuple) 
        #Doing padding as part of the dataset class. Get rid of this line below when the dataset only contains the gene names
        #self.data_tuple[0][idx] = torch.nn.functional.pad(X, (0, 0, 0, data_prep.max_length - X.shape[0]), value=FOLDSEEK_MISSING_IDX)
        #return tuple(torch.tensor(data[idx]) for data in self.data_tuple)


os.environ['OPENBLAS_NUM_THREADS'] = '1'

FOLDSEEK_MISSING_IDX = 20

with open("data/therapeutic_target_data/gene_aa_dict.pkl", "rb") as saved_file:
    gene_aa_dict = pickle.load(saved_file)
    

def run_finetune(embed, plm_type, celltype_dict, celltype_protein_dict, positive_proteins, negative_proteins, data_split_path, random_state, models_output_dir, embed_name, embed_name2, hparams, batch_size, num_epoch, train_size, val_size, weigh_sample, weigh_loss):

    # Training and validation
    X_train, X_test, context_train, context_test, y_train, y_test, groups_train, cts_train, groups_test = process_and_split_data(embed, positive_proteins, negative_proteins, celltype_protein_dict, celltype_dict, data_split_path, random_state=random_state, test_size=1-train_size-val_size)
    clf, pooling_clf, best_epoch, best_val_auprc, train_ranks, val_ranks = finetune_train_stage(X_train, context_train, y_train, plm_type, random_state, groups_train, cts_train, hparams, train_size, val_size, num_epoch, batch_size, weigh_sample, weigh_loss, models_output_dir, embed_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    positive_proportion_train = {}
    positive_proportion_train['celltype'] = sum(y_train) / len(y_train)
    wandb.log({f'train positive proportion celltype': positive_proportion_train['celltype'], 'best_val_auprc': best_val_auprc})

    ap5_cf, auroc_cf, protein_embeddings, genes, prot_celltypes, labels = finetune_evaluate(celltype_protein_dict, clf, pooling_clf, X_test, context_test, y_test, groups_test, models_output_dir, embed_name, train_ranks, val_ranks, device)
    ap5_20, auroc_1, auroc_10, auroc_20, auroc_cell, ap_cell, ap_5_cell, precision_5_cell, recall_5_cell, ref_embeddings, ref_celltypes = finetune_evaluate_by_cell(celltype_protein_dict, clf, pooling_clf, X_test, context_test, y_test, groups_test, models_output_dir, embed_name, train_ranks, val_ranks, device)

    metrics_dict = {}
    metrics_dict["ap5_20"] = ap5_20
    metrics_dict["auroc_1"] = auroc_1
    metrics_dict["auroc_10"] = auroc_10
    metrics_dict["auroc_20"] = auroc_20
    metrics_dict["ap5_cf"] = ap5_cf
    metrics_dict["auroc_cf"] = auroc_cf

    adata_prot_embs = store_prot_embs(protein_embeddings, genes, prot_celltypes, labels)
    adata_ref_embs = store_ref_embs(ref_embeddings, ref_celltypes)


    save_path = os.path.join(models_output_dir, f"{embed_name}_model.pt")
    save_path2 = os.path.join(models_output_dir, f"{embed_name2}_model.pt")

    #torch.save({'epoch': best_epoch, 'model_state_dict': clf.state_dict(), 'pooling_state_dict': pooling_clf.state_dict()}, save_path)
    torch.save({'epoch': best_epoch, 'model_state_dict': clf.state_dict()}, save_path)
    torch.save({'epoch': best_epoch, 'pooling_state_dict': pooling_clf.state_dict()}, save_path2)

    #return positive_proportion_train, positive_proportion_test, apr_5, apr_10, apr_15, apr_20, auroc_scores, ap_scores, test_recall_5, test_precision_5, test_ap_5, test_recall_10, test_precision_10, test_ap_10
    return metrics_dict, auroc_cell, ap_cell, ap_5_cell, precision_5_cell, recall_5_cell, adata_prot_embs, adata_ref_embs


def finetune_train_stage(X_train, context_train, y_train, plm_type, random_state, groups_train, cts_train, hparams, train_size, val_size, num_epoch, batch_size, weigh_sample, weigh_loss, models_output_dir, embed_name):
    #if not isinstance(X_train, torch.Tensor):
        #X_train = torch.from_numpy(X_train)
    
    if not isinstance(context_train, torch.Tensor):
        context_train = torch.from_numpy(context_train)

    split_state = 0
    n_splits = int((train_size+val_size)/val_size)
    #train_indices, val_indices = list(StratifiedGroupKFold(n_splits=n_splits, random_state=random_state, shuffle=True).split(X=X_train, groups=groups_train, y=y_train))[np.random.randint(0, n_splits)]  # borrow CV generator to generate one split
    train_indices, val_indices = list(StratifiedGroupKFold(n_splits=n_splits, random_state=split_state, shuffle=True).split(X=X_train, groups=groups_train, y=y_train))[random_state % n_splits]

    
    X_train_training = [X_train[index] for index in train_indices]
    X_train_val = [X_train[index] for index in val_indices]
    clf, pooling_clf, best_train_y, best_train_preds, best_train_cts, best_train_groups, cts_map_train, groups_map_train, best_val_y, best_val_preds, best_val_cts, best_val_groups, cts_map_val, groups_map_val, best_epoch, best_val_auprc = training_and_validation(X_train_training, X_train_val, context_train[train_indices], context_train[val_indices], torch.Tensor(y_train)[train_indices], torch.Tensor(y_train)[val_indices], np.array(cts_train)[train_indices], np.array(cts_train)[val_indices], np.array(groups_train)[train_indices], np.array(groups_train)[val_indices], plm_type, num_epoch, batch_size, weigh_sample, weigh_loss, hparams)
    #clf, pooling_clf, best_train_y, best_train_preds, best_train_cts, best_train_groups, cts_map_train, groups_map_train, best_val_y, best_val_preds, best_val_cts, best_val_groups, cts_map_val, groups_map_val, best_epoch, best_val_auprc = training_and_validation(X_train[train_indices], X_train[val_indices], context_train[train_indices], context_train[val_indices], torch.Tensor(y_train)[train_indices], torch.Tensor(y_train)[val_indices], np.array(cts_train)[train_indices], np.array(cts_train)[val_indices], np.array(groups_train)[train_indices], np.array(groups_train)[val_indices], num_epoch, batch_size, weigh_sample, weigh_loss, hparams, pooling)
    train_ranks, val_ranks = save_torch_train_val_preds(best_train_y, best_train_preds, best_train_groups, best_train_cts, best_val_y, best_val_preds, best_val_groups, best_val_cts, groups_map_train, groups_map_val, cts_map_train, cts_map_val, models_output_dir, embed_name, wandb)

    clf = clf.to(torch.device('cpu'))
    pooling_clf = pooling_clf.to(torch.device('cpu'))
    return clf, pooling_clf, best_epoch, best_val_auprc, train_ranks, val_ranks


def store_prot_embs(embeddings, genes, celltypes, labels):

    obs_dict = {}
    obs_dict["genes"] = genes
    obs_dict["celltypes"] = celltypes
    obs_dict["labels"] = labels

    obs_df = pd.DataFrame(obs_dict)
    
    #obsm = {}
    #obsm["embeddings"] = embeddings

    adata_object = ad.AnnData(X=embeddings, obs=obs_df)
    return adata_object


def store_ref_embs(ref_embs, celltypes):

    obs_dict = {}
    obs_dict["celltypes"] = celltypes

    obs_df = pd.DataFrame(obs_dict)

    adata_object = ad.AnnData(X=ref_embs, obs=obs_df)
    return adata_object


def genes_to_embeddings(test_data, max_length, plm_type):

    data_folder = "data/therapeutic_target_data"

    if plm_type == "esm2_8m":
        tokenized_file = os.path.join(data_folder, "ESM_esm2_t6_8M_UR50D_features.h5")
    elif plm_type == "esm2_650m":
        tokenized_file = os.path.join(data_folder, "ESM_esm2_t33_650M_UR50D_features.h5")
    elif plm_type == "progen2_small":
        tokenized_file = os.path.join(data_folder, "ProGen_progen2-small_features.h5")
    elif plm_type == "progen2_large":
        tokenized_file = os.path.join(data_folder, "ProGen_progen2-large_features.h5")
    else:
        raise ValueError(f"Unknown plm_type: {plm_type}")

    h5file = h5py.File(tokenized_file, 'r')

    embeddings = []
    for gene in test_data:
        aa_sequence = gene_aa_dict[gene]
        h5dataset = h5file[aa_sequence]  
        esm_embed = torch.tensor(np.array(h5dataset))
        padded_embed = torch.nn.functional.pad(esm_embed, (0, 0, 0, max_length - esm_embed.shape[0]), value=FOLDSEEK_MISSING_IDX)
        embeddings.append(padded_embed)
    embed_tensor = torch.stack(embeddings)
    return embed_tensor

def get_max_length(test_data):
    test_max_length = 0
    for gene in test_data:
        aa_sequence = gene_aa_dict[gene]
        sequence_length = len(aa_sequence)
        if sequence_length > test_max_length:
            test_max_length = sequence_length
    return test_max_length


def finetune_evaluate(celltype_protein_dict, clf, pooling_clf, X_test, context_test, y_test, groups_test, plm_type, models_output_dir, embed_name, train_ranks, val_ranks, device):

    pooling_clf.eval()
    clf.eval()

    X_test_all = []
    context_test_all = []
    y_test_all = []
    celltype_all = []
    all_preds = torch.tensor([])
    all_embeddings = []

    for cell in X_test.keys():
        for gene in X_test[cell]:
            X_test_all.append(gene)
            celltype_all.append(cell)
        for context in context_test[cell]:
            context_test_all.append(context)
        for y in y_test[cell]:
            y_test_all.append(y)
        
    y_test_all = torch.tensor(y_test_all)
    context_test_all = torch.stack(context_test_all)

    batch_size = 32
    test_dataset = CustomDataset(X_test_all, context_test_all, y_test_all)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    pooling_clf = pooling_clf.to(device)
    clf = clf.to(device)

    with torch.no_grad():
        for X, context, y in test_loader:
            batch_max = get_max_length(X)  
            padded_X = genes_to_embeddings(X, batch_max, plm_type)  
            
            padded_X, context, y = padded_X.to(device), context.to(device), y.to(device)

            mask = (padded_X != FOLDSEEK_MISSING_IDX)[:, :, 0] 
            pooled_emb, ref_emb = pooling_clf(padded_X, context, mask)

            for embedding in pooled_emb:
                all_embeddings.append(embedding)

            y_pred = torch.sigmoid(clf(pooled_emb)).squeeze(-1)
            #y_pred = y_pred.detach().cpu().numpy()
            y_pred = y_pred.detach().cpu()
            all_preds = torch.cat([all_preds, y_pred])
    
    y_test_all = y_test_all.detach().cpu().numpy()
    all_preds = all_preds.detach().cpu().numpy()
    all_embeddings = torch.stack(all_embeddings)
    all_embeddings = all_embeddings.detach().cpu().numpy()

    ap5_cf, auroc_cf = get_context_free_metrics(y_test_all, all_preds)
    return ap5_cf, auroc_cf, all_embeddings, X_test_all, celltype_all, y_test_all

    
    




def finetune_evaluate_by_cell(celltype_protein_dict, clf, pooling_clf, X_test, context_test, y_test, groups_test, plm_type, models_output_dir, embed_name, train_ranks, val_ranks, device):
    
    ap_5 = {}
    ap_10 = {}
    ap_15 = {}
    ap_20 = {}
    auroc_scores = {}
    ap_scores = {}
    positive_proportion_test = {}
    test_recall_5 = {}
    test_precision_5 = {}
    test_recall_10 = {}
    test_precision_10 = {}

    test_ranks = {}
    size_dict = {}

    ref_embeddings = []
    all_celltypes = []
    used_celltypes = []
    
    pooling_clf = pooling_clf.to(device)
    clf = clf.to(device)


    for celltype in celltype_protein_dict:
        if celltype not in X_test: continue
        clf.eval()
        pooling_clf.eval()
        all_preds = torch.tensor([])

        batch_size = 32
        test_dataset = CustomDataset(X_test[celltype], context_test[celltype])
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        with torch.no_grad():
            for X, context in test_loader:
                
                #new_X_test = X_test_to_tensor(X_test[celltype])    #If the initial data contains the ESM embeddings
                test_max = get_max_length(X)                   #If the initial data contains the gene names
                new_X_test = genes_to_embeddings(X, test_max, plm_type)  #If the initial data contains the gene names
                mask = (new_X_test != FOLDSEEK_MISSING_IDX)[:, :, 0]
                new_X_test = new_X_test.to(device)
                mask = mask.to(device)
                context = context.to(device)

                pooled_emb, ref_emb = pooling_clf(new_X_test, context, mask)
                print("Ref Emb Shape: ", ref_emb.shape)
                first_ref_emb = ref_emb[0]
                if celltype not in used_celltypes:
                    for embedding in first_ref_emb:
                        ref_embeddings.append(embedding)
                        all_celltypes.append(celltype)
                    used_celltypes.append(celltype)

                y_pred = torch.sigmoid(clf(pooled_emb)).squeeze(-1)
                y_pred = y_pred.detach().cpu()
                all_preds = torch.cat([all_preds, y_pred])

        #print("All preds shape: ", all_preds.shape)
        size_dict[celltype] = all_preds.shape[0]
        all_preds = all_preds.numpy()
        # Evaluation on test set
        ap_5[celltype], ap_10[celltype], ap_15[celltype], ap_20[celltype], auroc_scores[celltype], ap_scores[celltype], test_recall_5[celltype], test_precision_5[celltype], test_recall_10[celltype], test_precision_10[celltype], sorted_y_test, sorted_preds_test, sorted_groups_test, positive_proportion_test[celltype] = get_metrics(y_test, all_preds, groups_test, celltype)
                
    ref_embeddings = torch.stack(ref_embeddings)
    ref_embeddings = ref_embeddings.detach().cpu().numpy()
    #print("Final Ref Embed Shape: ", ref_embeddings.shape)

    ap5_20, auroc_1, auroc_10, auroc_20 = get_context_specific_metrics(ap_5, auroc_scores, size_dict)
    #ap5_cf, auroc_cf = get_context_free_metrics(y_pred, y_test)

    return ap5_20, auroc_1, auroc_10, auroc_20, auroc_scores, ap_scores, ap_5, test_precision_5, test_recall_5, ref_embeddings, all_celltypes







def set_seed(seed):

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def run_one_seed(seed, args):

    set_seed(seed)
    hparams_raw = get_hparams(args)
    run = wandb.init(config = hparams_raw, project = "finetune", entity = "sahilp113-duke-university", name=f"seed_{seed}", group=f"{args.task_name}_seeds", reinit=True)
    hparams = wandb.config
    args.random_state = seed

    metrics, auroc_cell, ap_cell, ap_5_cell, precision_5_cell, recall_5_cell, adata_prot_embs, adata_ref_embs, metrics_output_dir = run_all(args, hparams, wandb)

    run.finish()
    return metrics, auroc_cell, ap_cell, ap_5_cell, precision_5_cell, recall_5_cell, adata_prot_embs, adata_ref_embs, metrics_output_dir



def run_all(args, hparams, wandb):
    print(args)

    # Set up model environment and data/model paths
    models_output_dir, metrics_output_dir, random_state, embed_path, labels_path = setup_paths(args)
    
    # Load data
    positive_proteins_prefix = args.positive_proteins_prefix + "_" + args.tt_disease
    negative_proteins_prefix = args.negative_proteins_prefix + "_" + args.tt_disease
    embed, celltype_dict, celltype_protein_dict, positive_proteins, negative_proteins, _ = load_data(embed_path, labels_path, positive_proteins_prefix, negative_proteins_prefix, None, args.task_name)
    print("Finished reading data, evaluating...\n")

    # Run model
    data_split_path = args.data_split_path + "_" + args.tt_disease + ".json"
    #positive_proportion_train, positive_proportion_test, apr_5, apr_10, apr_15, apr_20, auroc_scores, ap_scores, test_recall_5, test_precision_5, test_ap_5, test_recall_10, test_precision_10, test_ap_10 = run_finetune(embed, celltype_dict, celltype_protein_dict, positive_proteins, negative_proteins, data_split_path, random_state, models_output_dir, args.embed, args.embed2, hparams, args.batch_size, args.num_epoch, args.train_size, args.val_size, args.weigh_sample, args.weigh_loss)
    metrics_dict, auroc_cell, ap_cell, ap_5_cell, precision_5_cell, recall_5_cell, adata_prot_embs, adata_ref_embs = run_finetune(embed, args.plm, celltype_dict, celltype_protein_dict, positive_proteins, negative_proteins, data_split_path, random_state, models_output_dir, args.embed, args.embed2, hparams, args.batch_size, args.num_epoch, args.train_size, args.val_size, args.weigh_sample, args.weigh_loss)
 
    wandb.log({f"summary/{k}": v for k, v in metrics_dict.items() if k != "seed"})
    
    return metrics_dict, auroc_cell, ap_cell, ap_5_cell, precision_5_cell, recall_5_cell, adata_prot_embs, adata_ref_embs, metrics_output_dir
    #return positive_proportion_train, positive_proportion_test, apr_5, apr_10, apr_15, apr_20, auroc_scores, ap_scores, test_recall_5, test_precision_5, test_ap_5, test_recall_10, test_precision_10, test_ap_10

    

def main():
    args = create_parser()

    all_metrics_dict = {}
    metrics_output_dir = ""
    auroc_cell_dict = {}
    ap_cell_dict = {}
    ap_5_cell_dict = {}
    precision_5_cell_dict = {}
    recall_5_cell_dict = {}
    #adata_prot_embs = ad.AnnData()
    #adata_ref_embs = ad.AnnData()

    n_seeds = args.n_seeds
    seeds = list(range(n_seeds))
    for seed in seeds:
        metrics_dict, auroc_cell, ap_cell, ap_5_cell, precision_5_cell, recall_5_cell, adata_prot_embs, adata_ref_embs, metrics_output_dir = run_one_seed(seed, args)
        
        if seed == 0:
            auroc_cell_dict = auroc_cell
            ap_cell_dict = ap_cell
            ap_5_cell_dict = ap_5_cell
            precision_5_cell_dict = precision_5_cell
            recall_5_cell_dict = recall_5_cell

            adata_prot_path = os.path.join(metrics_output_dir, "prot_embs.h5ad")
            adata_prot_embs.write_h5ad(adata_prot_path)

            adata_ref_path = os.path.join(metrics_output_dir, "ref_embs.h5ad")
            adata_ref_embs.write_h5ad(adata_ref_path)

        
        for metric in metrics_dict.keys():
            if metric not in all_metrics_dict.keys():
                all_metrics_dict[metric] = []
            all_metrics_dict[metric].append(metrics_dict[metric])

    mean_dict = {}
    std_dev_dict = {}

    for metric in all_metrics_dict.keys():
        mean_dict[metric] = np.mean(all_metrics_dict[metric])
        std_dev_dict[metric] = np.std(all_metrics_dict[metric])

        # Generate outputs and plots
    print("Finished evaluation, generating plots...\n")
    output_results_path = os.path.join(metrics_output_dir, f"{args.embed}_{args.task_name}_results.csv")  #Chnaged from results to results2 to results3 create a new results file
    output_figs_path = os.path.join(metrics_output_dir, f"{args.embed}_{args.task_name}_")

    combined_dict = {}
    combined_dict['means'] = mean_dict
    combined_dict['std_devs'] = std_dev_dict
    combined_dict['auroc_per_cell'] = auroc_cell_dict
    combined_dict['ap_per_cell'] = ap_cell_dict
    combined_dict['ap_5_per_cell'] = ap_5_cell_dict
    combined_dict['precision_5_per_cell'] = precision_5_cell_dict
    combined_dict['recall_5_per_cell'] = recall_5_cell_dict


    with open(output_results_path, 'w') as f:
        json.dump(combined_dict, f) 
     

    agg = wandb.init(
        project="finetune",
        entity="sahilp113-duke-university",
        name=f"{args.task_name}_aggregate",
        group=f"{args.task_name}_seeds",
        reinit=True,
    )  

    for metric in mean_dict.keys():
        mean_label = "agg/mean_" + metric
        std_dev_label = "agg/std_" + metric
        wandb.log({
            mean_label: mean_dict[metric],
            std_dev_label: std_dev_dict[metric]
        })

    #wandb.log({
    #    "agg/n_seeds": len(seeds),
    #    "agg/split_seed": args.split_seed,
    #})

    agg.finish()

    #save_results(output_results_path, apr_5, apr_10, apr_15, apr_20, ap_scores, auroc_scores, test_ap_5, test_recall_5, test_precision_5, test_ap_10, test_recall_10, test_precision_10)




if __name__ == '__main__':
    main()


    


