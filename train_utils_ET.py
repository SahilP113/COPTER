# Training and validation helper functions


import wandb
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from torch.utils.data import Dataset
import torch.nn as nn

from metrics.metrics_utils_TCR import get_metrics, precision_recall_at_k

from models.mlp_model import MLP
from copy import deepcopy
from models.swe_pooling import SWE_Pooling
import pickle
import h5py


FOLDSEEK_MISSING_IDX = 20

class CustomDataset(Dataset):
    def __init__(self, *data_tuple):
        self.data_tuple = data_tuple
    
    def __len__(self):
        return len(self.data_tuple[0])

    def __getitem__(self, idx):
        X = self.data_tuple[0][idx]
        return tuple(data[idx] for data in self.data_tuple) 


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
        tcr_h5dataset = tcr_h5file[tcr_sequence]  
        esm_embed = torch.tensor(np.array(tcr_h5dataset))
        padded_embed = torch.nn.functional.pad(esm_embed, (0, 0, 0, batch_max - esm_embed.shape[0]), value=FOLDSEEK_MISSING_IDX)
        tcr_embeddings.append(padded_embed)
    tcr_embed_tensor = torch.stack(tcr_embeddings)
    return tcr_embed_tensor


#Getting the max length of the TCR sequence in every batch
def get_max_length(tcr_batch):
    batch_max_length = 0
    for tcr_sequence in tcr_batch:
        sequence_length = len(tcr_sequence)
        if sequence_length > batch_max_length:
            batch_max_length = sequence_length
    return batch_max_length

#The context is now a 320 dimensional vector, not a 64 dimensional vector, so set L to 320.
def get_epitope(epitope_batch, plm_type):

    data_folder = "pan_prep_data"

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



#def training_and_validation(X_train, X_val, context_train, context_val, y_train, y_val, cts_train, cts_val, groups_train, groups_val, num_epoch, batch_size, weigh_sample, weigh_loss, hparams, no_val=False):
def training_and_validation(TCR_train, TCR_val, epitope_train, epitope_val, label_train, label_val, plm_type, num_epoch, batch_size, weigh_sample, weigh_loss, hparams, no_val=False):
        
    if plm_type == "esm2_8m":
        L = 320
        D = 320
        d_cell = 320
    elif plm_type == "esm2_650m":
        L = 1280
        D = 1280
        d_cell = 1280
    elif plm_type == "progen2_small":
        L = 1024
        D = 1024
        d_cell = 1024
    elif plm_type == "progen2_large":
        L = 2560
        D = 2560
        d_cell = 2560
    else:
        raise ValueError(f"Unknown plm_type: {plm_type}")

    freeze_swe = False
    dim_out = 1

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
    best_val_auprc = 0

    train_dataset = CustomDataset(TCR_train, epitope_train, label_train.unsqueeze(-1))
    if not no_val:
        val_dataset = CustomDataset(TCR_val, epitope_val, label_val.unsqueeze(-1))

    sampler = None
    shuffle = True
    if weigh_sample:
        class_sample_num = torch.unique(label_train, return_counts=True)[1]
        weights = torch.DoubleTensor([1/class_sample_num[y.int().item()] for y in label_train])
        sampler = WeightedRandomSampler(weights, len(weights))
        shuffle = False
    drop_last = False
    print("Batch Size: ", batch_size)
    if batch_size is None:
        batch_size = len(train_dataset)
    print("Batch Size New: ", batch_size)
    if (hparams[norm] == "bn" or hparams[norm] == "ln") and len(train_dataset) % batch_size < 3:
        drop_last = True

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, shuffle=shuffle, num_workers=2, drop_last=drop_last)
    if not no_val:
        val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False, num_workers=2)  # set val batch size to full-batch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    pos_weight = None
    if weigh_loss:
        pos_weight = torch.Tensor([(label_train.shape[0] - label_train.sum().item()) / label_train.sum().item()]).to(device)
    #Add coefficients to BCE loss and center loss
    loss_func = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    #Add pooling parameters to Adam optimizer
    optim = torch.optim.Adam(list(model.parameters()) + list(pooling.parameters()), lr=hparams[lr], weight_decay=hparams[wd])
    
    wandb.watch(model, log_freq=20)
    for i in range(num_epoch):
        print(f"Epoch {i+1}\n---------------")
        _, _, train_y, train_preds = train_epoch(model, pooling, train_loader, plm_type, optim, loss_func, batch_size, wandb, device)
        if not no_val:
            _, val_auprc, val_y, val_preds = validate_epoch(model, pooling, val_loader, plm_type, loss_func, wandb, device)
            if val_auprc > best_val_auprc:
                clf = deepcopy(model)
                pooling_clf = deepcopy(pooling)
                best_val_auprc = val_auprc
                best_epoch = i
                best_val_y = val_y.copy()
                best_val_preds = val_preds.copy()
                best_train_y = train_y.copy()
                best_train_preds = train_preds.copy()

    if no_val:
        best_train_y = train_y
        best_train_preds = train_preds
        return model, pooling, best_train_y, best_train_preds
    
    return clf, pooling_clf, best_train_y, best_train_preds, best_val_y, best_val_preds, best_epoch, best_val_auprc



def train_epoch(model, pooling, train_loader, plm_type, optim, loss_func, batch_size, wandb, device):

    model.train()
    train_size = len(train_loader.dataset)
    total_sample = total_loss = 0
    all_y = torch.tensor([])
    all_preds = torch.tensor([])
    for i, (X, context, y) in enumerate(train_loader):

        #if i == 0:
            #print("X: ", X)
        all_y = torch.cat([all_y, y])
        batch_max = get_max_length(X)
        padded_tcr = get_padded_tcr(X, batch_max, plm_type)
        epitope = get_epitope(context, plm_type)

        padded_tcr, epitope, y = padded_tcr.to(device), epitope.to(device), y.to(device)
        optim.zero_grad()

        mask = (padded_tcr != FOLDSEEK_MISSING_IDX)[:, :, 0]

        pooled_X = pooling(padded_tcr, epitope, mask)

        preds = model(pooled_X)
        loss = loss_func(preds, y.float())

        #Backpropagate over aggregate loss 
        loss.backward()
        optim.step()

        all_preds = torch.cat([all_preds, preds.cpu()])
        total_sample += batch_size
        total_loss += float(loss) * batch_size

        if i % 20 == 0:
            loss, current = loss.item(), i * len(X)
            print(f"train loss: {loss:.4f} [{current}/{train_size}]")
            wandb.log({f"train loss":loss})

    print("Finished with batches...")

    all_y = all_y.detach().numpy().astype(int)
    all_preds = torch.sigmoid(all_preds).detach().numpy()
    
    train_accuracy, train_f1_score, train_apr_5, train_apr_10, train_apr_15, train_apr_20, train_auroc, train_auprc, train_bal_acc, train_recall_5, train_precision_5, train_ap_5, train_recall_10, train_precision_10, train_ap_10, _, _, _, _ = get_metrics(all_y, all_preds, "training")
    #train_apr_5, train_apr_10, train_apr_15, train_apr_20, train_auroc, train_auprc, train_recall_5, train_precision_5, train_ap_5, train_recall_10, train_precision_10, train_ap_10, _, _, _, _ = get_metrics(all_y, all_preds, all_groups, "training")

    total_loss = total_loss / total_sample
    wandb.log({f"train APR@5": train_apr_5,
               f"train APR@10": train_apr_10,
               f"train APR@15": train_apr_15,
               f"train APR@20": train_apr_20,
               f"train AUPRC": train_auprc,
               f"train AUROC": train_auroc,
               f"train accuracy": train_accuracy,
               f"train F1-score": train_f1_score,
               f"train balanced accuracy": train_bal_acc,
               f"train recall@5": train_recall_5,
               f"train recall@10": train_recall_10,
               f"train precision@5": train_precision_5,
               f"train precision@10": train_precision_10,
               f"train AP@5": train_ap_5,
               f"train AP@10": train_ap_10})

    print("Finished with one full epoch...")

    return total_loss, train_auprc, all_y, all_preds


@torch.no_grad()
def validate_epoch(model, pooling, val_loader, plm_type, loss_func, wandb, device):
    val_size = len(val_loader.dataset)

    model.eval()
    val_loss = 0
    all_y = torch.tensor([])
    all_preds = torch.tensor([])
    
    for X, context, y in val_loader:
        all_y = torch.cat([all_y, y])

        batch_max = get_max_length(X)
        padded_tcr = get_padded_tcr(X, batch_max, plm_type)
        epitope = get_epitope(context, plm_type)

        padded_tcr, epitope, y = padded_tcr.to(device), epitope.to(device), y.to(device)

        mask = (padded_tcr != FOLDSEEK_MISSING_IDX)[:, :, 0]

        pooled_X = pooling(padded_tcr, epitope, mask)

        preds = model(pooled_X)
        all_preds = torch.cat([all_preds, preds.cpu()])
        #val_loss += loss_func(preds, y).item() * X.shape[0]        #Use this line when the initial data contains the embeddings
        val_loss += loss_func(preds, y.float()).item() * padded_tcr.shape[0]  #Use this line when the initial data contains gene names

    print("Finished all batches in validation...")
    
    val_loss /= val_size

    ys, preds = all_y.detach().numpy(), torch.sigmoid(all_preds).detach().numpy()

    val_accuracy, val_f1_score, val_apr_5, val_apr_10, val_apr_15, val_apr_20, val_auroc, val_auprc, val_bal_acc, val_recall_5, val_precision_5, val_ap_5, val_recall_10, val_precision_10, val_ap_10, _, _, _, _ = get_metrics(ys, preds, "training")
    #val_apr_5, val_apr_10, val_apr_15, val_apr_20, val_auroc, val_auprc, val_recall_5, val_precision_5, val_ap_5, val_recall_10, val_precision_10, val_ap_10, _, _, _, _ = get_metrics(ys, preds, groups, "training")

    wandb.log({f"val loss":val_loss,
               f"val APR@5":val_apr_5,
               f"val APR@10":val_apr_10,
               f"val APR@15":val_apr_15,
               f"val APR@20":val_apr_20,
               f"val AUPRC":val_auprc,
               f"val AUROC":val_auroc,
               f"val accuracy":val_accuracy,
               f"val F1-score":val_f1_score,
               f"val balanced accuracy":val_bal_acc,
               f"val recall@5":val_recall_5,
               f"val recall@10":val_recall_10,
               f"val precision@5":val_precision_5,
               f"val precision@10":val_precision_10,
               f"val AP@5":val_ap_5,
               f"val AP@10":val_ap_10})

    print("Finished with calculating metrics...")
    return val_loss, val_auprc, ys, preds
