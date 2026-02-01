import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
import requests
from sklearn.model_selection import train_test_split
import torch
import h5py
import ast
import scanpy as sc
import anndata as ad
from scipy.io import mmread




def get_ens_aa_dict(all_genes_df):

    ensembl_aa_dict = {} 

    for i, gene in enumerate(all_genes_df["gene_id"]):
        aa_sequence = all_genes_df["aa_sequence"][i]
        ensembl_aa_dict[gene] = aa_sequence
    return ensembl_aa_dict


def get_gene_aa_dict(gene_ens_dict, ensembl_aa_dict):
    gene_aa_dict = {}

    for gene in gene_ens_dict.keys():
        ens_id = gene_ens_dict[gene]
        if ens_id in ensembl_aa_dict.keys():
            gene_aa_dict[gene] = ensembl_aa_dict[ens_id]

    return gene_aa_dict



def get_train_data(adata, adata_control, splits_data, prefix, num_de_genes):

    train_genes = []
    train_controls = []
    train_y = []
    train_de_idxs = []

    np.random.seed(42)

    control_array = adata_control.X.A
    de_genes = adata.uns['rank_genes_groups_cov_all']

    for perturbation in splits_data["train"]:
        if perturbation != "ctrl":
            #dict_name = "A549_" + perturbation + "_1+1"
            #dict_name = "rpe1_" + perturbation + "_1+1"
            #dict_name = "K562_" + perturbation + "_1+1"
            dict_name = prefix + "_" + perturbation + "_1+1"
            de_idx = np.where(adata.var_names.isin(np.array(de_genes[dict_name][:num_de_genes])))[0]
            de_idx = torch.tensor(de_idx)

            gene = perturbation.split("+")[0]
            perturb_data = adata[adata.obs["condition"] == perturbation]
            for i in range(perturb_data.n_obs):
                train_genes.append(gene)
                rand_int = np.random.randint(0, adata_control.n_obs)
                random_control = torch.tensor(control_array[rand_int])
                train_controls.append(random_control)
                train_de_idxs.append(de_idx)
                
            perturb_array = perturb_data.X.A
            for i in range(perturb_array.shape[0]):
                perturb_tensor = torch.tensor(perturb_array[i])
                train_y.append(perturb_tensor)

        else:
            de_idx = [-1] * num_de_genes
            de_idx = torch.tensor(de_idx)
            
            for i in range(adata_control.n_obs):
                control_exp = torch.tensor(control_array[i])
                train_genes.append("control")
                train_controls.append(control_exp)
                train_y.append(control_exp)
                train_de_idxs.append(de_idx)

    train_controls = torch.stack(train_controls)
    train_y = torch.stack(train_y)
    train_de_idxs = torch.stack(train_de_idxs)

    return train_genes, train_controls, train_y, train_de_idxs
    



def get_val_data(adata, adata_control, splits_data, prefix, num_de_genes):

    val_genes = []
    val_controls = []
    val_y = []
    val_de_idxs = []

    np.random.seed(42)

    control_array = adata_control.X.A
    de_genes = adata.uns['rank_genes_groups_cov_all']

    for perturbation in splits_data["val"]:
        if perturbation != "ctrl":
            #dict_name = "A549_" + perturbation + "_1+1"
            #dict_name = "rpe1_" + perturbation + "_1+1"
            #dict_name = "K562_" + perturbation + "_1+1"
            dict_name = prefix + "_" + perturbation + "_1+1"
            de_idx = np.where(adata.var_names.isin(np.array(de_genes[dict_name][:num_de_genes])))[0]
            de_idx = torch.tensor(de_idx)
            
            gene = perturbation.split("+")[0]
            perturb_data = adata[adata.obs["condition"] == perturbation]
            for i in range(perturb_data.n_obs):
                val_genes.append(gene)
                rand_int = np.random.randint(0, adata_control.n_obs)
                random_control = torch.tensor(control_array[rand_int])
                val_controls.append(random_control)
                val_de_idxs.append(de_idx)
                
            perturb_array = perturb_data.X.A
            for i in range(perturb_array.shape[0]):
                perturb_tensor = torch.tensor(perturb_array[i])
                val_y.append(perturb_tensor)

        else:
            de_idx = [-1] * num_de_genes
            de_idx = torch.tensor(de_idx)
            
            for i in range(adata_control.n_obs):
                control_exp = torch.tensor(control_array[i])
                val_genes.append("control")
                val_controls.append(control_exp)
                val_y.append(control_exp)
                val_de_idxs.append(de_idx)

    val_controls = torch.stack(val_controls)
    val_y = torch.stack(val_y)
    val_de_idxs = torch.stack(val_de_idxs)

    return val_genes, val_controls, val_y, val_de_idxs





def get_test_data(adata, adata_control, splits_data, prefix, num_de_genes):

    test_genes = []
    test_controls = []
    test_y = []
    test_de_idxs = []

    np.random.seed(42)

    control_array = adata_control.X.A
    de_genes = adata.uns['rank_genes_groups_cov_all']

    for perturbation in splits_data["test"]:
        if perturbation != "ctrl":
            #dict_name = "A549_" + perturbation + "_1+1"
            #dict_name = "rpe1_" + perturbation + "_1+1"
            #dict_name = "K562_" + perturbation + "_1+1"
            dict_name = prefix + "_" + perturbation + "_1+1"
            de_idx = np.where(adata.var_names.isin(np.array(de_genes[dict_name][:num_de_genes])))[0]
            de_idx = torch.tensor(de_idx)

            gene = perturbation.split("+")[0]
            perturb_data = adata[adata.obs["condition"] == perturbation]
            for i in range(perturb_data.n_obs):
                test_genes.append(gene)
                rand_int = np.random.randint(0, adata_control.n_obs)
                random_control = torch.tensor(control_array[rand_int])
                test_controls.append(random_control)
                test_de_idxs.append(de_idx)
                
            perturb_array = perturb_data.X.A
            for i in range(perturb_array.shape[0]):
                perturb_tensor = torch.tensor(perturb_array[i])
                test_y.append(perturb_tensor)

        else:
            de_idx = [-1] * num_de_genes
            de_idx = torch.tensor(de_idx)

            for i in range(adata_control.n_obs):
                control_exp = torch.tensor(control_array[i])
                test_genes.append("control")
                test_controls.append(control_exp)
                test_y.append(control_exp)
                test_de_idxs.append(de_idx)

    test_controls = torch.stack(test_controls)
    test_y = torch.stack(test_y)
    test_de_idxs = torch.stack(test_de_idxs)

    return test_genes, test_controls, test_y, test_de_idxs



def get_all_data(data_file_path, gene_ens_path, ensembl_aa_path, splits_file_path, task_name):
    adata_perturb = sc.read_h5ad(data_file_path)
    all_genes_df = pd.read_csv(ensembl_aa_path)
    num_de_genes = 20

    if task_name == "replogle_rpe1":
        prefix = "rpe1"
    elif task_name == "replogle_k562":
        prefix = "K562"

    with open(splits_file_path, "rb") as splits_file:
        splits_data = pickle.load(splits_file)
    
    with open(gene_ens_path, "rb") as genes_file:
        gene_ens_dict = pickle.load(genes_file)

    ensembl_aa_dict = get_ens_aa_dict(all_genes_df)
    gene_aa_dict = get_gene_aa_dict(gene_ens_dict, ensembl_aa_dict)

    adata_control = adata_perturb[adata_perturb.obs["condition"] == "ctrl"]
    train_genes, train_controls, train_y, train_de_idxs = get_train_data(adata_perturb, adata_control, splits_data, prefix, num_de_genes)
    val_genes, val_controls, val_y, val_de_idxs = get_val_data(adata_perturb, adata_control, splits_data, prefix, num_de_genes)
    test_genes, test_controls, test_y, test_de_idxs = get_test_data(adata_perturb, adata_control, splits_data, prefix, num_de_genes)

    return gene_aa_dict, train_genes, train_controls, train_y, train_de_idxs, val_genes, val_controls, val_y, val_de_idxs, test_genes, test_controls, test_y, test_de_idxs








