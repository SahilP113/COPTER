# Training and validation helper functions


import wandb
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch.nn as nn

from metrics.metrics_utils_perturb import calculate_mse20, compute_metrics
from models.mlp_model import MLP
from copy import deepcopy
from models.swe_pooling import SWE_Pooling
import pickle
import h5py
import os

#from inference_perturb import evaluate
#from inference_perturb import compute_metrics



FOLDSEEK_MISSING_IDX = 20

#Use this variable when the dataset contains the gene names. Otherwise, get it from data_prep.py

class CustomDataset(Dataset):
    def __init__(self, *data_tuple):
        self.data_tuple = data_tuple
    
    def __len__(self):
        return len(self.data_tuple[0])

    def __getitem__(self, idx):
        X = self.data_tuple[0][idx]
        return tuple(data[idx] for data in self.data_tuple) 


def get_esm_embeddings(gene_batch, gene_aa_dict, batch_max, plm_type, task_name):
    embeddings = []

    task_folder = os.path.join("data/gene_perturb_data", task_name)

    if plm_type == "esm2_8m":
        embed_size = 320
        embeddings_file = "ESM_esm2_t6_8M_UR50D_features.h5"
    elif plm_type == "esm2_650m":
        embed_size = 1280
        embeddings_file = "ESM_esm2_t33_650M_UR50D_features.h5"
    elif plm_type == "progen2_small":
        embed_size = 1024
        embeddings_file = "ProGen_progen2-small_features.h5"
    elif plm_type == "progen2_large":
        embed_size = 2560
        embeddings_file = "ProGen_progen2-large_features.h5"
    else:
        raise ValueError(f"Unknown plm_type: {plm_type}")

    tokenized_file = os.path.join(task_folder, embeddings_file)
    h5file = h5py.File(tokenized_file, 'r')

    for gene in gene_batch:
        if gene == 'control':
            control_embed = torch.zeros((1, embed_size))
            padded_embed = torch.nn.functional.pad(control_embed, (0, 0, 0, batch_max - control_embed.shape[0]), value=FOLDSEEK_MISSING_IDX)
            embeddings.append(padded_embed) 
        else:
            aa_sequence = gene_aa_dict[gene]
            h5dataset = h5file[aa_sequence]
            esm_embed = torch.tensor(np.array(h5dataset))
            padded_embed = torch.nn.functional.pad(esm_embed, (0, 0, 0, batch_max - esm_embed.shape[0]), value=FOLDSEEK_MISSING_IDX)
            embeddings.append(padded_embed)
    embed_tensor = torch.stack(embeddings)
    return embed_tensor

#Padding within every batch to reduce the size of the padding
def get_max_length(gene_batch, gene_aa_dict):
    batch_max_length = 0
    for gene in gene_batch:
        if gene != 'control':
            aa_sequence = gene_aa_dict[gene]
            sequence_length = len(aa_sequence)
            if sequence_length > batch_max_length:
                batch_max_length = sequence_length
    return batch_max_length


def train_val_test(gene_aa_dict, train_loader, val_loader, test_loader, plm_type, task_name, models_output_dir, embed_name, embed_name2, num_epoch, batch_size, weigh_sample, weigh_loss, hparams, no_val=False):
    #Train SWE and MLP here
    L = 320
    M = 100
    
    if plm_type == "esm2_8m":
        D = 320
    elif plm_type == "esm2_650m":
        D = 1280
    elif plm_type == "progen2_small":
        D = 1024
    elif plm_type == "progen2_large":
        D = 2560
    else:
        raise ValueError(f"Unknown plm_type: {plm_type}")


    d_cell = 5000 #size of context vector
    dim_out = 5000
    freeze_swe = False

    """
    Train an MLP on a train-val split.
    """
    norm = "norm"
    actn = "actn"
    hidden_dim_1 = "hidden_dim_1"
    hidden_dim_2 = "hidden_dim_2"
    hidden_dim_3 = "hidden_dim_3"
    dropout = "dropout"
    lr = "lr"
    wd = "wd"
    order = "order"
    min_val = 100000



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Cuda Version:", torch.version.cuda)  


    if hparams[hidden_dim_2] == 0:
        hidden_dims = [hparams[hidden_dim_1]]
    elif hparams[hidden_dim_3] == 0:
        hidden_dims = [hparams[hidden_dim_1], hparams[hidden_dim_2]]
    else:
        hidden_dims = [hparams[hidden_dim_1], hparams[hidden_dim_2], hparams[hidden_dim_3]]

    #initialize pooling here
    pooling = SWE_Pooling(d_in=D, num_slices=L, num_ref_points=M, d_cell=d_cell, freeze_swe=freeze_swe)
    model = MLP(in_dim = L, hidden_dims = hidden_dims, final_dim = dim_out, p = hparams[dropout], norm=hparams[norm], actn=hparams[actn], order=hparams[order])
    pooling = pooling.to(device)
    model = model.to(device)

    if torch.cuda.is_available():
        print("Running on GPU: ", torch.cuda.get_device_name(0))
    else:
        print("Running on CPU")
    
    
    pos_weight = None
    if weigh_loss:
        pos_weight = torch.Tensor([(y_train.shape[0] - y_train.sum().item()) / y_train.sum().item()]).to(device)
    #loss_func = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    loss_func = nn.MSELoss()
    #Add pooling parameters to Adam optimizer
    optim = torch.optim.Adam(list(model.parameters()) + list(pooling.parameters()), lr=hparams[lr], weight_decay=hparams[wd])
    
    wandb.watch(model, log_freq=20)
    for epoch in range(num_epoch):
        print(f"Epoch {epoch+1}\n---------------")
        pooling.train()
        model.train()
        train_size = len(train_loader.dataset)
        total_sample = total_loss = 0


        for i, (gene, context, y, de_idx) in enumerate(train_loader):

            batch_max = get_max_length(gene, gene_aa_dict)
            X = get_esm_embeddings(gene, gene_aa_dict, batch_max, plm_type, task_name)
            X = torch.nn.functional.normalize(X, dim=-1)
            
            delta = torch.sub(y, context)

            X, context, delta = X.to(device), context.to(device), delta.to(device)
            optim.zero_grad()

            mask = (X != FOLDSEEK_MISSING_IDX)[:, :, 0]

            pooled_X = pooling(X, context, mask)
            #print("pooled X shape:", pooled_X.shape)

            preds = model(pooled_X)
            loss = loss_func(preds, delta)

            #Backpropagate over aggregate loss 
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(pooling.parameters(), 1.0)

            optim.step()

            total_sample += batch_size
            total_loss += float(loss) * batch_size

            if i % 20 == 0:
                loss, current = loss.item(), i * len(X)
                print(f"train loss: {loss:.4f} [{current}/{train_size}]")
                wandb.log({f"train loss":loss})

        print("Finished with batches...")

        train_res = evaluate(gene_aa_dict, train_loader, pooling, model, plm_type, task_name, device)
        train_metrics, _ = compute_metrics(train_res)
        
        #total_loss = total_loss / total_sample

        val_res = evaluate(gene_aa_dict, val_loader, pooling, model, plm_type, task_name, device)
        val_metrics, _ = compute_metrics(val_res)


        print(f"Epoch {epoch + 1}: Train Overall MSE: {train_metrics['mse']:.4f} "
              f"Validation Overall MSE: {val_metrics['mse']:.4f}.")
        
        # Print epoch performance for DE genes
        print(f"Train Top 20 DE MSE: {train_metrics['mse_de']:.4f} "
              f"Validation Top 20 DE MSE: {val_metrics['mse_de']:.4f}.")
        

        metrics = ['mse', 'pearson']
        for m in metrics:
            wandb.log({'train_' + m: train_metrics[m],
                    'val_'+m: val_metrics[m],
                    'train_de_' + m: train_metrics[m + '_de'],
                    'val_de_'+m: val_metrics[m + '_de']})
        
        if val_metrics['mse_de'] < min_val:
            best_epoch = epoch
            min_val = val_metrics['mse_de']
            best_mlp = deepcopy(model)
            best_pooling = deepcopy(pooling)

    test_res = evaluate(gene_aa_dict, test_loader, best_pooling, best_mlp, plm_type, task_name, device)
    test_metrics, test_pert_res = compute_metrics(test_res)

    print(f"Best performing model: Test Top 20 DE MSE: {test_metrics['mse_de']:.4f}")
        
    metrics = ['mse', 'pearson']
    for m in metrics:
        wandb.log({'test_' + m: test_metrics[m],
                   'test_de_'+m: test_metrics[m + '_de']                     
                })
    
    save_path = os.path.join(models_output_dir, f"{embed_name}_model.pt")
    save_path2 = os.path.join(models_output_dir, f"{embed_name2}_model.pt")

    torch.save({'epoch': best_epoch, 'model_state_dict': best_mlp.state_dict()}, save_path)
    torch.save({'epoch': best_epoch, 'pooling_state_dict': best_pooling.state_dict()}, save_path2)

    return test_metrics


def evaluate(gene_aa_dict, loader, pooling_model, mlp_model, plm_type, task_name, device):
    """
    Run model in inference mode using a given data loader
    """
    pooling_model.eval()
    mlp_model.eval()

    pooling_model.to(device)
    mlp_model.to(device)
    pert_cat = []
    pred = []
    truth = []
    pred_de = []
    truth_de = []
    results = {}
    logvar = []
    
    for itr, (gene, context, y, de_idx) in enumerate(loader):

        batch_max = get_max_length(gene, gene_aa_dict)
        X = get_esm_embeddings(gene, gene_aa_dict, batch_max, plm_type, task_name)
        X = torch.nn.functional.normalize(X, dim=-1)
        mask = (X != FOLDSEEK_MISSING_IDX)[:, :, 0]

        t = torch.sub(y, context)
        X, context, t, de_idx, mask = X.to(device), context.to(device), t.to(device), de_idx.to(device), mask.to(device)
        pert_cat.extend(gene)

        with torch.no_grad():

            pooled_X = pooling_model(X, context, mask)
            p = mlp_model(pooled_X) #Size: [64, 5000]
            pred.extend(p.cpu())
            truth.extend(t.cpu())
            
            # Differentially expressed genes
            for itr, de_idx2 in enumerate(de_idx):
                pred_de.append(p[itr, de_idx2])
                truth_de.append(t[itr, de_idx2])

    # all genes
    results['pert_cat'] = np.array(pert_cat)
    pred = torch.stack(pred)
    truth = torch.stack(truth)
    results['pred']= pred.detach().cpu().numpy()
    results['truth']= truth.detach().cpu().numpy()

    pred_de = torch.stack(pred_de)
    truth_de = torch.stack(truth_de)
    results['pred_de']= pred_de.detach().cpu().numpy()
    results['truth_de']= truth_de.detach().cpu().numpy()
        
    return results

