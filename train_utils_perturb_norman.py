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



def get_esm_embeddings(gene_batch, gene_aa_dict, batch_max, plm_type):
    embeddings = []
   
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

    tokenized_file = os.path.join("data/gene_perturb_data/norman", embeddings_file)
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


def train_val_test(gene_aa_dict, train_loader, val_loader, test_loader, plm_type, models_output_dir, embed_name, embed_name2, num_epoch, batch_size, weigh_sample, weigh_loss, hparams, no_val=False):
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

    d_cell = 5045 #5000  #size of context vector
    dim_out = 5045 #5000 
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


        for i, (perturb_batch, context_batch, y, de_idx) in enumerate(train_loader):
            
            optim.zero_grad()

            two_genes_mask = []

            for perturbation in perturb_batch:
                if "+" in perturbation:
                    p = perturbation.split("+")
                    if p[0] != "ctrl" and p[1] != "ctrl":
                        two_genes_mask.append(True)
                    else:
                        two_genes_mask.append(False)
                else:
                    two_genes_mask.append(False)
            two_genes_mask = torch.tensor(two_genes_mask)
            one_gene_mask = ~two_genes_mask


            perturb_two_gene = [p for p, m in zip(perturb_batch, two_genes_mask) if m]
            context_two_gene = context_batch[two_genes_mask]
            y_two_gene = y[two_genes_mask]
            de_idx_two_gene = de_idx[two_genes_mask]

            two_gene_batch_size = len(perturb_two_gene)
            two_gene_loss = train(perturb_two_gene, context_two_gene, y_two_gene, de_idx_two_gene, gene_aa_dict, pooling, model, plm_type, optim, loss_func, device, two_genes=True)

            perturb_one_gene = [p for p, m in zip(perturb_batch, one_gene_mask) if m]
            context_one_gene = context_batch[one_gene_mask]
            y_one_gene = y[one_gene_mask]
            de_idx_one_gene = de_idx[one_gene_mask]

            one_gene_batch_size = len(perturb_one_gene)
            one_gene_loss = train(perturb_one_gene, context_one_gene, y_one_gene, de_idx_one_gene, gene_aa_dict, pooling, model, plm_type, optim, loss_func, device, two_genes=False)

            loss = (two_gene_batch_size/batch_size)*two_gene_loss + (one_gene_batch_size/batch_size)*one_gene_loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(pooling.parameters(), 1.0)

            optim.step()


            total_sample += batch_size
            total_loss += float(loss) * batch_size

            if i % 20 == 0:
                loss, current = loss.item(), i * len(perturb_batch)
                print(f"train loss: {loss:.4f} [{current}/{train_size}]")
                wandb.log({f"train loss":loss})

        print("Finished with batches...")

        train_res = evaluate(gene_aa_dict, train_loader, pooling, model, plm_type, device)
        train_metrics, _ = compute_metrics(train_res)
        
        #total_loss = total_loss / total_sample

        val_res = evaluate(gene_aa_dict, val_loader, pooling, model, plm_type, device)
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

    test_res = evaluate(gene_aa_dict, test_loader, best_pooling, best_mlp, plm_type, device)
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



def train(perturb_batch, context_batch, y, de_idx, gene_aa_dict, pooling, model, optim, plm_type, loss_func, device, two_genes=True):

    pooled_X = torch.zeros(2, 3)

    if two_genes:
        first_genes = []
        second_genes = []
        for perturbation in perturb_batch:
            genes = perturbation.split("+")
            first_genes.append(genes[0])
            second_genes.append(genes[1])

        pooled_emb1 = get_pooled_embedding(first_genes, context_batch, gene_aa_dict, pooling, plm_type, device)
        #print("Emb 1 Shape: ", pooled_emb1.shape)
        pooled_emb2 = get_pooled_embedding(second_genes, context_batch, gene_aa_dict, pooling, plm_type, device)
        #print("Emb 2 Shape: ", pooled_emb2.shape)
        pooled_X = pooled_emb1 + pooled_emb2
        #print("Pooled X Shape (2 genes): ", pooled_X.shape)

    else:
        
        all_genes = []

        for perturbation in perturb_batch:
            if perturbation == "control":
                all_genes.append("control")
            else:
                genes = perturbation.split("+")
                if genes[0] == "ctrl":
                    all_genes.append(genes[1])
                if genes[1] == "ctrl":
                    all_genes.append(genes[0])
        

        pooled_X = get_pooled_embedding(all_genes, context_batch, gene_aa_dict, pooling, plm_type, device)
        #pooled_X = get_pooled_embedding(perturb_batch, context_batch, gene_aa_dict, pooling, device)
        print("Pooled X Shape (1 gene): ", pooled_X.shape)


    delta = torch.sub(y, context_batch)
    delta = delta.to(device)
    preds = model(pooled_X)

    loss = loss_func(preds, delta)

    return loss



def get_pooled_embedding(gene, context, gene_aa_dict, pooling, plm_type, device):

    batch_max = get_max_length(gene, gene_aa_dict)
    X = get_esm_embeddings(gene, gene_aa_dict, batch_max, plm_type)
    X = torch.nn.functional.normalize(X, dim=-1)

    X, context = X.to(device), context.to(device)
    mask = (X != FOLDSEEK_MISSING_IDX)[:, :, 0]
    pooled_emb = pooling(X, context, mask)
    return pooled_emb


def evaluate(gene_aa_dict, loader, pooling_model, mlp_model, plm_type, device):
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

    with torch.no_grad():
    
        for itr, (perturb_batch, context_batch, y, de_idx) in enumerate(loader):

            two_genes_mask = []

            for perturbation in perturb_batch:
                if "+" in perturbation:
                    p = perturbation.split("+")
                    if p[0] != "ctrl" and p[1] != "ctrl":
                        two_genes_mask.append(True)
                    else:
                        two_genes_mask.append(False)
                else:
                    two_genes_mask.append(False)
            two_genes_mask = torch.tensor(two_genes_mask)
            one_gene_mask = ~two_genes_mask

            print("de_idx shape: ", de_idx.shape)
            print("one_gene_mask shape: ", one_gene_mask.shape)
            print("two_gene_mask shape: ", two_genes_mask.shape)

            perturb_two_gene = [p for p, m in zip(perturb_batch, two_genes_mask) if m]
            context_two_gene = context_batch[two_genes_mask]
            y_two_gene = y[two_genes_mask]
            de_idx_two_gene = de_idx[two_genes_mask]
            
            if len(perturb_two_gene) > 0:
                p_two_gene, t_two_gene = evaluate2(perturb_two_gene, context_two_gene, y_two_gene, de_idx_two_gene, gene_aa_dict, pooling_model, mlp_model, plm_type, device, two_genes=True)

                p_two_gene = p_two_gene.detach().cpu()
                t_two_gene = t_two_gene.detach().cpu()

                pert_cat.extend(perturb_two_gene)
                pred.extend(p_two_gene)
                truth.extend(t_two_gene)

                for itr, de_idx2 in enumerate(de_idx_two_gene):
                    pred_de.append(p_two_gene[itr, de_idx2])
                    truth_de.append(t_two_gene[itr, de_idx2])


            perturb_one_gene = [p for p, m in zip(perturb_batch, one_gene_mask) if m]
            context_one_gene = context_batch[one_gene_mask]
            y_one_gene = y[one_gene_mask]
            de_idx_one_gene = de_idx[one_gene_mask]

            if len(perturb_one_gene) > 0:
                p_one_gene, t_one_gene = evaluate2(perturb_one_gene, context_one_gene, y_one_gene, de_idx_one_gene, gene_aa_dict, pooling_model, mlp_model, plm_type, device, two_genes=False)
                p_one_gene = p_one_gene.detach().cpu()
                t_one_gene = t_one_gene.detach().cpu()
                
                pred.extend(p_one_gene)
                truth.extend(t_one_gene)
                pert_cat.extend(perturb_one_gene)
            
                for itr, de_idx2 in enumerate(de_idx_one_gene):
                    pred_de.append(p_one_gene[itr, de_idx2])
                    truth_de.append(t_one_gene[itr, de_idx2]) 
           
                
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



def evaluate2(perturb_batch, context_batch, y, de_idx, gene_aa_dict, pooling_model, mlp_model, plm_type, device, two_genes):


    if two_genes:
        first_genes = []
        second_genes = []
        for perturbation in perturb_batch:
            genes = perturbation.split("+")
            first_genes.append(genes[0])
            second_genes.append(genes[1])

        pooled_emb1 = get_pooled_embedding(first_genes, context_batch, gene_aa_dict, pooling_model, plm_type, device)
        pooled_emb2 = get_pooled_embedding(second_genes, context_batch, gene_aa_dict, pooling_model, plm_type, device)
        pooled_X = (pooled_emb1 + pooled_emb2)/2

    
    else:
        all_genes = []
        for i, perturbation in enumerate(perturb_batch):
            #print("i: ", i)
            if perturbation == "control":
                all_genes.append("control")
                print("control")
            else:
                genes = perturbation.split("+")
                if genes[0] == "ctrl":
                    all_genes.append(genes[1])
                    print("p1: " + genes[0] + ", p2: " + genes[1])

                elif genes[1] == "ctrl":
                    all_genes.append(genes[0])
                    print("p1: " + genes[0] + ", p2: " + genes[1])

        
        pooled_X = get_pooled_embedding(all_genes, context_batch, gene_aa_dict, pooling_model, plm_type, device)
    

    t = torch.sub(y, context_batch)
    t, de_idx = t.to(device), de_idx.to(device)

    p = mlp_model(pooled_X) #Size: [64, 5000]

    
    return p, t





