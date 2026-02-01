# Training and validation helper functions

import wandb
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from torch.utils.data import Dataset
import torch.nn as nn

from metrics.metrics_utils_tt import get_metrics, precision_recall_at_k
from models.mlp_model import MLP
from copy import deepcopy
from models.tt_swe_pooling import SWE_Pooling
import pickle
import h5py



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

with open("data/therapeutic_target_data/gene_aa_dict.pkl", "rb") as saved_file:
    gene_aa_dict = pickle.load(saved_file)


def get_esm(gene):
    if gene in gene_aa_dict.keys():
        aa_sequence = gene_aa_dict[gene]
        h5dataset = h5file[aa_sequence]  
        esm_embed = torch.tensor(np.array(h5dataset))
        return esm_embed
    else:
        #dummy_embed = torch.zeros((400, 1280))
        dummy_embed = torch.zeros((400, 320))
        return dummy_embed


#Need to get max_length for padding
#Need to figure out what the batch looks like
def get_padded_esm(gene_batch, batch_max, plm_type):

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
    for gene in gene_batch:
        if gene in gene_aa_dict.keys():
            aa_sequence = gene_aa_dict[gene]
            h5dataset = h5file[aa_sequence]  
            esm_embed = torch.tensor(np.array(h5dataset))
            padded_embed = torch.nn.functional.pad(esm_embed, (0, 0, 0, batch_max - esm_embed.shape[0]), value=FOLDSEEK_MISSING_IDX)
            embeddings.append(padded_embed)
        else:
            dummy_embed = torch.zeros((400, 320))
            #dummy_embed = torch.zeros((400, 1280))
            pad_dum_embed = torch.nn.functional.pad(dummy_embed, (0, 0, 0, batch_max - dummy_embed.shape[0]), value=FOLDSEEK_MISSING_IDX)
            embeddings.append(pad_dum_embed)
    embed_tensor = torch.stack(embeddings)
    return embed_tensor

#Padding within every batch to reduce the size of the padding
def get_max_length(gene_batch):
    batch_max_length = 0
    for gene in gene_batch:
        aa_sequence = gene_aa_dict[gene]
        sequence_length = len(aa_sequence)
        if sequence_length > batch_max_length:
            batch_max_length = sequence_length
    return batch_max_length




def training_and_validation(X_train, X_val, context_train, context_val, y_train, y_val, cts_train, cts_val, groups_train, groups_val, plm_type, num_epoch, batch_size, weigh_sample, weigh_loss, hparams, no_val=False):
    #Train SWE and MLP here
    #Xtrain, Xval will be ESM token embeddings
    L = 64 
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


    d_cell = 64
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

    if not no_val:
        cts_map_val = np.unique(cts_val, return_inverse=True)[0]  # factorize cts_val
        cts_val = np.unique(cts_val, return_inverse=True)[1]
        groups_map_val = np.unique(groups_val, return_inverse=True)[0]  # factorize groups_val
        groups_val = np.unique(groups_val, return_inverse=True)[1]
    
    cts_map_train = np.unique(cts_train, return_inverse=True)[0]  # factorize cts_train
    cts_train = np.unique(cts_train, return_inverse=True)[1]
    groups_map_train = np.unique(groups_train, return_inverse=True)[0]  # factorize groups_train
    groups_train = np.unique(groups_train, return_inverse=True)[1]

    #train_dataset = TensorDataset(X_train, context_train, y_train.unsqueeze(-1), torch.from_numpy(cts_train), torch.from_numpy(groups_train))
    train_dataset = CustomDataset(X_train, context_train, y_train.unsqueeze(-1), torch.from_numpy(cts_train), torch.from_numpy(groups_train))
    if not no_val:
        #val_dataset = TensorDataset(X_val, context_val, y_val.unsqueeze(-1), torch.from_numpy(cts_val), torch.from_numpy(groups_val))
        val_dataset = CustomDataset(X_val, context_val, y_val.unsqueeze(-1), torch.from_numpy(cts_val), torch.from_numpy(groups_val))

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

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, shuffle=shuffle, num_workers=2, drop_last=drop_last)
    if not no_val:
        #val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False, num_workers=2)  # set val batch size to full-batch
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)  # set val batch size to full-batch
        print("Val Dataset Size: ", len(val_dataset))

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
    # X_train.shape[1] = 64 and L = 64, so change X_train.shape[1] to L
    # model = MLP(in_dim = X_train.shape[1], hidden_dims = hidden_dims, p = hparams[dropout], norm=hparams[norm], actn=hparams[actn], order=hparams[order])
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
    loss_func = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    #Add pooling parameters to Adam optimizer
    optim = torch.optim.Adam(list(model.parameters()) + list(pooling.parameters()), lr=hparams[lr], weight_decay=hparams[wd])
    

    wandb.watch(model, log_freq=20)
    for i in range(num_epoch):
        print(f"Epoch {i+1}\n---------------")
        _, _, train_y, train_preds, train_cts, train_groups = train_epoch(model, pooling, train_loader, plm_type, optim, loss_func, batch_size, wandb, device)
        if not no_val:
            _, val_auprc, val_y, val_preds, val_cts, val_groups = validate_epoch(model, pooling, val_loader, plm_type, loss_func, wandb, device)
            print("Done with validating epoch")
            if val_auprc > best_val_auprc:
                clf = deepcopy(model)
                pooling_clf = deepcopy(pooling)
                best_val_auprc = val_auprc
                best_epoch = i
                best_val_groups = val_groups.copy().astype(int)
                best_val_y = val_y.copy()
                best_val_preds = val_preds.copy()
                best_train_groups = train_groups.copy().astype(int)
                best_train_y = train_y.copy()
                best_train_preds = train_preds.copy()
                best_val_cts = val_cts.copy().astype(int)
                best_train_cts = train_cts.copy().astype(int)
                print("Ready for second epoch")

    if no_val:
        best_train_groups = train_groups.astype(int)
        best_train_y = train_y
        best_train_preds = train_preds
        best_train_cts = train_cts.astype(int)
        return model, pooling, best_train_y, best_train_preds, best_train_cts, best_train_groups, cts_map_train, groups_map_train
    
    return clf, pooling_clf, best_train_y, best_train_preds, best_train_cts, best_train_groups, cts_map_train, groups_map_train, best_val_y, best_val_preds, best_val_cts, best_val_groups, cts_map_val, groups_map_val, best_epoch, best_val_auprc


def train_epoch(model, pooling, train_loader, plm_type, optim, loss_func, batch_size, wandb, device):
    model.train()
    pooling.train()

    train_size = len(train_loader.dataset)
    total_sample = total_loss = 0
    all_y = torch.tensor([])
    all_preds = torch.tensor([])
    all_cts = torch.tensor([])
    all_groups = torch.tensor([])
    for i, (X, context, y, cts, groups) in enumerate(train_loader):
 
        #if i == 0:
            #print("X: ", X)
        all_y = torch.cat([all_y, y])
        all_cts = torch.cat([all_cts, cts])
        all_groups = torch.cat([all_groups, groups])

        #Uncomment the 6 lines below when the initial dataset only contains the gene names
        batch_max = get_max_length(X)
        padded_X = get_padded_esm(X, batch_max, plm_type)
        print("Padded train X shape: ", padded_X.shape)
        
        padded_X, context, y = padded_X.to(device), context.to(device), y.to(device)
        optim.zero_grad()

        mask = (padded_X != FOLDSEEK_MISSING_IDX)[:, :, 0]
        pooled_X, ref_embs = pooling(padded_X, context, mask)

        preds = model(pooled_X)      #Apply SWE_pooling here also
        #preds = model(X)
        loss = loss_func(preds, y)
        
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
    all_cts = all_cts.detach().numpy().astype(int)
    all_groups = all_groups.detach().numpy().astype(int)

    train_ap_5, train_ap_10, train_ap_15, train_ap_20, train_auroc, train_auprc, train_recall_5, train_precision_5, train_recall_10, train_precision_10, _, _, _, _ = get_metrics(all_y, all_preds, all_groups, "training")

    total_loss = total_loss / total_sample
    wandb.log({f"train AP@5": train_ap_5,
               f"train AP@10": train_ap_10,
               f"train AP@15": train_ap_15,
               f"train AP@20": train_ap_20,
               f"train AUPRC": train_auprc,
               f"train AUROC": train_auroc,
               f"train recall@5": train_recall_5,
               f"train recall@10": train_recall_10,
               f"train precision@5": train_precision_5,
               f"train precision@10": train_precision_10})

    print("Finished with one full epoch...")

    return total_loss, train_auprc, all_y, all_preds, all_cts, all_groups


def validate_epoch(model, pooling, val_loader, plm_type, loss_func, wandb, device):
    print("In validation")
    val_size = len(val_loader.dataset)

    model.eval()
    pooling.eval()

    val_loss = 0
    all_y = torch.tensor([])
    all_preds = torch.tensor([])
    all_cts = torch.tensor([])
    all_groups = torch.tensor([])
    
    with torch.no_grad():

        for X, context, y, cts, groups in val_loader:
            print("In val loader")
            all_y = torch.cat([all_y, y])
            #X, context, y = X.to(device), context.to(device), y.to(device)
            all_cts = torch.cat([all_cts, cts])
            all_groups = torch.cat([all_groups, groups])
            print("X shape: ", np.shape(X))


            batch_max = get_max_length(X)
            print("Batch Max: ", batch_max)
            padded_X = get_padded_esm(X, batch_max, plm_type)
            print("Padded X shape: ", padded_X.shape)
            #print("Got ESM embedding")
            padded_X, context, y = padded_X.to(device), context.to(device), y.to(device)
            #print("Before Mask")
            mask = (padded_X != FOLDSEEK_MISSING_IDX)[:, :, 0]
            print("Mask shape: ", mask.shape)
         
            pooled_X, ref_embs = pooling(padded_X, context, mask)
        
            print("Finished pooling")


            preds = model(pooled_X)
        
            all_preds = torch.cat([all_preds, preds.cpu()])
            #val_loss += loss_func(preds, y).item() * X.shape[0]        #Use this line when the initial data contains the embeddings
            val_loss += loss_func(preds, y).item() * padded_X.shape[0]  #Use this line when the initial data contains gene names
            print("Finished with this batch")

    print("Finished all batches in validation...")
    
    val_loss /= val_size

    ys, preds, cts, groups = all_y.detach().numpy(), torch.sigmoid(all_preds).detach().numpy(), all_cts.detach().numpy(), all_groups.detach().numpy()

    val_ap_5, val_ap_10, val_ap_15, val_ap_20, val_auroc, val_auprc, val_recall_5, val_precision_5, val_recall_10, val_precision_10, _, _, _, _ = get_metrics(ys, preds, groups, "training")

    wandb.log({f"val loss":val_loss,
               f"val AP@5":val_ap_5,
               f"val AP@10":val_ap_10,
               f"val AP@15":val_ap_15,
               f"val AP@20":val_ap_20,
               f"val AUPRC":val_auprc,
               f"val AUROC":val_auroc,
               f"val recall@5":val_recall_5,
               f"val recall@10":val_recall_10,
               f"val precision@5":val_precision_5,
               f"val precision@10":val_precision_10})

    print("Finished with calculating metrics...")
    return val_loss, val_auprc, ys, preds, cts, groups
