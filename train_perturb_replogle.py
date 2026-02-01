import pandas as pd
import numpy as np
import os, wandb, random
import json

from setup import create_parser, get_hparams, setup_paths
from train_utils_perturb_replogle import train_val_test
from metrics.metrics_utils_perturb import calculate_mse20, save_results
from data_prep.data_prep_replogle import get_all_data

#import os
#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

import torch
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from torch.utils.data import Dataset

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils import shuffle
import pickle
import h5py

os.environ['OPENBLAS_NUM_THREADS'] = '1'

FOLDSEEK_MISSING_IDX = 20

class CustomDataset(Dataset):
    def __init__(self, *data_tuple):
        self.data_tuple = data_tuple
    
    def __len__(self):
        return len(self.data_tuple[0])

    def __getitem__(self, idx):
        X = self.data_tuple[0][idx]
        return tuple(data[idx] for data in self.data_tuple) 



def run_finetune(gene_aa_dict, train_genes, train_controls, train_y, train_de_idxs, val_genes, val_controls, val_y, val_de_idxs, test_genes, test_controls, test_y, test_de_idxs, models_output_dir, plm_type, task_name, embed_name, embed_name2, hparams, batch_size, num_epoch, weigh_sample, weigh_loss):

    norm = "norm"
    sampler = None
    shuffle = True
    if weigh_sample:
        class_sample_num = torch.unique(y_train, return_counts=True)[1]
        weights = torch.DoubleTensor([1/class_sample_num[y.int().item()] for y in y_train])
        sampler = WeightedRandomSampler(weights, len(weights))
        shuffle = False
    drop_last = False
    print("Batch Size: ", batch_size)
    if batch_size is None:
        batch_size = len(train_dataset)
    print("Batch Size New: ", batch_size)
    if (hparams[norm] == "bn" or hparams[norm] == "ln") and len(train_dataset) % batch_size < 3:
        drop_last = True

    train_dataset = CustomDataset(train_genes, train_controls, train_y, train_de_idxs)
    val_dataset = CustomDataset(val_genes, val_controls, val_y, val_de_idxs)
    test_dataset = CustomDataset(test_genes, test_controls, test_y, test_de_idxs)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, shuffle=shuffle, num_workers=2, drop_last=drop_last)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)  
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    test_metrics = train_val_test(gene_aa_dict, train_loader, val_loader, test_loader, plm_type, task_name, models_output_dir, embed_name, embed_name2, num_epoch, batch_size, weigh_sample, weigh_loss, hparams)

    return test_metrics




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

    metrics, metrics_output_dir = run_all(args, hparams, wandb)

    run.finish()
    return metrics, metrics_output_dir



def run_all(args, hparams, wandb):
    print(args)

    models_output_dir, metrics_output_dir, random_state, embed_path, labels_path = setup_paths(args)

    task_name = args.perturb_task
    task_folder = os.path.join("data/gene_perturb_data", task_name)
    splits_file = task_name + "_simulation_1_0.75.pkl"

    data_file_path = os.path.join(task_folder, "perturb_processed.h5ad")
    gene_ens_path = os.path.join(task_folder, "gene_ensembl_dict.pkl")
    ensembl_aa_path = os.path.join(task_folder, "perturbed_genes.csv")
    splits_file_path = os.path.join(task_folder, "splits", splits_file)

    gene_aa_dict, train_genes, train_controls, train_y, train_de_idxs, val_genes, val_controls, val_y, val_de_idxs, test_genes, test_controls, test_y, test_de_idxs = get_all_data(data_file_path, gene_ens_path, ensembl_aa_path, splits_file_path, task_name)

    data_split_path = args.data_split_path + ".json"

    test_metrics = run_finetune(gene_aa_dict, train_genes, train_controls, train_y, train_de_idxs, val_genes, val_controls, val_y, val_de_idxs, test_genes, test_controls, test_y, test_de_idxs, models_output_dir, args.plm, task_name, args.embed, args.embed2, hparams, args.batch_size, args.num_epoch, args.weigh_sample, args.weigh_loss)
    
    return test_metrics, metrics_output_dir

    


def main():
    args = create_parser()

    all_metrics_dict = {}
    metrics_output_dir = ""

    num_seeds = args.n_seeds
    seeds = list(range(num_seeds))
    for seed in seeds:
        metrics_dict, metrics_output_dir = run_one_seed(seed, args)
        
        for metric in metrics_dict.keys():
            if metric not in all_metrics_dict.keys():
                all_metrics_dict[metric] = []
            all_metrics_dict[metric].append(metrics_dict[metric])

    mean_dict = {}
    std_dev_dict = {}

    for metric in all_metrics_dict.keys():
        mean_dict[metric] = float(np.mean(all_metrics_dict[metric]))
        std_dev_dict[metric] = float(np.std(all_metrics_dict[metric]))

        # Generate outputs and plots
    print("Finished evaluation, generating plots...\n")
    output_results_path = os.path.join(metrics_output_dir, f"{args.embed}_{args.task_name}_results.csv")  #Chnaged from results to results2 to results3 create a new results file
    output_figs_path = os.path.join(metrics_output_dir, f"{args.embed}_{args.task_name}_")

    combined_dict = {}
    combined_dict['means'] = mean_dict
    combined_dict['std_devs'] = std_dev_dict

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

    agg.finish()      


if __name__ == '__main__':
    main()
