import argparse
from collections import Counter
from typing import Dict, List
import numpy as np
import pandas as pd
import json, matplotlib, os
import torch
from sklearn.model_selection import StratifiedGroupKFold
import networkx as nx
import pickle
import h5py

from read_data import load_data # load_PPI_data
from extract_txdata_utils import *

MAX_RETRY = 10  # To mitigate the effect of random state, we will redo data splitting for MAX_RETRY times if the number of positive samples in test set is less than TEST_CELLTYPE_POS_NUM_MIN
TEST_CELLTYPE_POS_NUM_MIN = 5 # For each cell type, the number of positive samples in test set must be greater than 5, or else the disease won't be evlauated
max_length = 0

with open("../therapeutic_target_data/gene_aa_dict.pkl", "rb") as saved_file:
    gene_aa_dict = pickle.load(saved_file)

with open("../therapeutic_target_data/cell_embeds.pkl", "rb") as saved_file:
    cell_embeds = pickle.load(saved_file)

"""
def read_args():
    parser = argparse.ArgumentParser()

    # PINNACLE pretrained representations
    parser.add_argument("--embeddings_dir", type=str, default="../therapeutic_target_data/pinnacle_embeds/")
    parser.add_argument("--embed", type=str, default="pinnacle")
    
    # Cell type specific PPI networks
    # parser.add_argument("--celltype_ppi", type=str, default="../data/networks/ppi_edgelists/", help="Filename (prefix) of cell type PPI.")
    
    # Fine-tuning data
    parser.add_argument('--positive_proteins_prefix', type=str, default="../therapeutic_target_data/therapeutic_target_task/positive_proteins")
    parser.add_argument('--negative_proteins_prefix', type=str, default="../therapeutic_target_data/therapeutic_target_task/negative_proteins")
    parser.add_argument('--raw_data_prefix', type=str, default="../therapeutic_target_data/therapeutic_target_task/raw_targets")

    # Parameters for data split size
    parser.add_argument("--train_size", type=float, default=0.6)
    parser.add_argument("--val_size", type=float, default=0.2)

    # Output
    parser.add_argument('--data_split_path', type=str, default="../raw_data/therapeutic_target_task/data_split")

    parser.add_argument("--random_state", type=int, default=1)
    args = parser.parse_args()

    return args
"""



def get_context(cell):
    cell_embed = cell_embeds[cell]
    return cell_embed

def process_and_split_data(embed, positive_proteins, negative_proteins, celltype_protein_dict, celltype_dict, data_split_path, random_state, test_size):
    """
    First generate data (averaging same protein embeddings in different celltypes for the downstream task.  Then split the data into train/test sets while grouping by protein and stratified by cell types.
    """
    L = 64 
    M = 100
    B = 32
    D = 320
    freeze_swe = False
    #Create a new embedding called new_embed
    #pooling = architectures.SWE_Pooling(d_in=D, num_slices=L, num_ref_points=M, freeze_swe=freeze_swe)

    esm_embed = {}  #Change this later
    cell_embed = {}
    test_list = ['acinar cell of salivary gland', 'adipocyte', 'adventitial cell', 'alveolar fibroblast', 'artery endothelial cell']


    
    #for cell in test_list:
    for cell in celltype_protein_dict:
        print(cell)
        cell_num = celltype_dict[cell]
        cell_context = get_context(cell)
        #temp_embeds = []
        sequence_lengths = []
        esm_embed[cell_num] = []
        cell_embed[cell_num] = []
        
        for protein in celltype_protein_dict[cell]:
          
            esm_embed[cell_num].append(protein)
            cell_embed[cell_num].append(cell_context)
         
        cell_embed[cell_num] = torch.stack(cell_embed[cell_num])
    


    pos_embed = []
    neg_embed = []
    pos_cell_embed = []
    neg_cell_embed = []
    pos_prots_strata = []  # Celltypes (needs to be stratified)
    neg_prots_strata = []
    pos_prots_group = []  # Protein (needs to be grouped and stay in the same data split)
    neg_prots_group = []

    # Generate data for split
    #for celltype in test_list:
    for celltype in celltype_dict:
        if celltype not in positive_proteins: continue
        pos_prots = positive_proteins[celltype]
        neg_prots = negative_proteins[celltype]

        pos_indices = np.where(np.isin(np.array(celltype_protein_dict[celltype]), np.unique(pos_prots)))[0]
        if len(pos_indices) == 0: continue
        assert len(pos_indices) == len(pos_prots)
        pos_list = [esm_embed[celltype_dict[celltype]][index] for index in pos_indices]
        pos_embed.append(pos_list)
       #pos_embed.append(esm_embed[celltype_dict[celltype]][pos_indices])
        pos_cell_embed.append(cell_embed[celltype_dict[celltype]][pos_indices])  #Added
        pos_prots_strata.extend([celltype] * len(pos_indices))
        pos_prots_group.extend(pos_prots)

        neg_indices = np.where(np.isin(np.array(celltype_protein_dict[celltype]), np.unique(neg_prots)))[0]
        assert len(neg_indices) != 0
        assert len(neg_indices) == len(neg_prots)
        neg_list = [esm_embed[celltype_dict[celltype]][index] for index in neg_indices]
        neg_embed.append(neg_list)
        #neg_embed.append(esm_embed[celltype_dict[celltype]][neg_indices])
        neg_cell_embed.append(cell_embed[celltype_dict[celltype]][neg_indices])  #Added
        neg_prots_strata.extend([celltype] * len(neg_indices))
        neg_prots_group.extend(neg_prots)

    #pos_embed = torch.tensor(np.concatenate(pos_embed, axis = 0)) #Might not convert this to a tensor here
    #neg_embed = torch.tensor(np.concatenate(neg_embed, axis = 0))
    
    pos_embed = [embedding for tensor_list in pos_embed for embedding in tensor_list]  #tensor_list is a list of gene names and embedding is a gene name
    neg_embed = [embedding for tensor_list in neg_embed for embedding in tensor_list]  
    pos_cell_embed = torch.tensor(np.concatenate(pos_cell_embed, axis = 0))
    neg_cell_embed = torch.tensor(np.concatenate(neg_cell_embed, axis = 0))

    assert len(pos_embed) == len(pos_prots_group)
    assert len(pos_prots_group) == len(pos_prots_strata)
    
    # Conduct train-test split in a grouped and stratified way, and also ensuring positive fraction by stratifying the positive and negative embeddings separately.
    print("Checking for...", data_split_path)
    if os.path.exists(data_split_path): # Generate new splits
        print("Data split file found. Loading data splits...")
        indices = json.load(open(data_split_path, "r"))
        pos_train_indices = torch.tensor(indices["pos_train_indices"])       #May need to copy and paste these four lines of code at the bottom
        pos_test_indices = torch.tensor(indices["pos_test_indices"])
        neg_train_indices = torch.tensor(indices["neg_train_indices"])
        neg_test_indices = torch.tensor(indices["neg_test_indices"])

        #May need to get rid of the data split file and generate my own splits
        #y_test = {celltype: np.concatenate([[1 for ind in pos_test_indices if pos_prots_strata[ind] == celltype], [0 for ind in neg_test_indices if neg_prots_strata[ind] == celltype]]) for celltype in test_list}

        y_test = {celltype: np.concatenate([[1 for ind in pos_test_indices if pos_prots_strata[ind] == celltype], [0 for ind in neg_test_indices if neg_prots_strata[ind] == celltype]]) for celltype in celltype_dict}
        print("Finished loading data splits.")

    else:

        print("Data split file not found. Generating data splits...")
        n_split = int(1/test_size)
        print("Number of splits...", n_split)
        
        def get_splits(cv):
            
            # Try all possible splits for positive examples
            for i, (pos_train_indices, pos_test_indices) in enumerate(cv.split(X=np.arange(len(pos_embed)), groups=pos_prots_group, y=pos_prots_strata)):
                #y_test = {celltype: [1 for ind in pos_test_indices if pos_prots_strata[ind] == celltype] for celltype in test_list}

                y_test = {celltype: [1 for ind in pos_test_indices if pos_prots_strata[ind] == celltype] for celltype in celltype_dict}
                if np.all(np.array(list(map(sum, y_test.values()))) > TEST_CELLTYPE_POS_NUM_MIN):
                    break
            
            # Randomly select train/test split for negative examples
            neg_train_indices, neg_test_indices = list(iter(cv.split(X=np.arange(len(neg_embed)), groups=neg_prots_group, y=neg_prots_strata)))[np.random.randint(0, n_split)]
            
            # Ensure that the test set has no overlap with train set
            assert np.all([Counter(pos_prots_group)[prot] == num for prot, num in Counter(np.array(pos_prots_group)[pos_test_indices]).items()])
            assert np.all([Counter(pos_prots_group)[prot] == num for prot, num in Counter(np.array(pos_prots_group)[pos_train_indices]).items()])
            
            # Combine test data
            #y_test = {celltype: np.concatenate([[1 for ind in pos_test_indices if pos_prots_strata[ind] == celltype], [0 for ind in neg_test_indices if neg_prots_strata[ind] == celltype]]) for celltype in test_list}

            y_test = {celltype: np.concatenate([[1 for ind in pos_test_indices if pos_prots_strata[ind] == celltype], [0 for ind in neg_test_indices if neg_prots_strata[ind] == celltype]]) for celltype in celltype_dict}
            return torch.tensor(pos_train_indices), torch.tensor(pos_test_indices), torch.tensor(neg_train_indices), torch.tensor(neg_test_indices), y_test
        
        try:
            cv = StratifiedGroupKFold(n_splits=n_split, random_state=random_state, shuffle=True)  # borrow this CV generator to generate one split of what we want
            pos_train_indices, pos_test_indices, neg_train_indices, neg_test_indices, y_test = get_splits(cv)
            print("First-try successful,", Counter(np.array(pos_prots_strata)[pos_train_indices]), Counter(np.array(pos_prots_strata)[pos_test_indices]))
        
        except:  # If failed to generate splits that contain valid number of pos/neg samples under our designated random_state, try with different random state for a few times more
            count = flag = 0
            while (count < MAX_RETRY and flag == 0):
                try:
                    new_random_state = np.random.randint(0, 100000)
                    cv = StratifiedGroupKFold(n_splits=n_split, random_state=new_random_state, shuffle=True)  # borrow this CV generator to generate one split of what we want
                    pos_train_indices, pos_test_indices, neg_train_indices, neg_test_indices, y_test = get_splits(cv)
                    print(("Re-tried successfully with new seed %s," % str(new_random_state)), Counter(np.array(pos_prots_strata)[pos_train_indices]), Counter(np.array(pos_prots_strata)[pos_test_indices]))
                    flag = 1
                except:
                    count += 1
                    continue

            if flag == 0:
                raise ValueError(f"Could not generate a valid train-test split. Number of positive test samples in some cell types is lower than {TEST_CELLTYPE_POS_NUM_MIN}.")

        indices_dict = {"pos_train_indices": pos_train_indices.tolist(),
                        "pos_test_indices": pos_test_indices.tolist(),
                        "neg_train_indices": neg_train_indices.tolist(),
                        "neg_test_indices": neg_test_indices.tolist()}
        print("Saving data splits to file", data_split_path)
        with open(data_split_path, "w") as outfile:
            json.dump(indices_dict, outfile)
        print("Finished saving data splits.")
    
    #Modify X_train here
    pos_train_embed = [pos_embed[index] for index in pos_train_indices]
    neg_train_embed = [neg_embed[index] for index in neg_train_indices]
    X_train = [embedding for embedding_list in [pos_train_embed, neg_train_embed] for embedding in embedding_list]  #Embedding is a gene name, embedding_list is a list of proteins
    #X_train = torch.cat([pos_train_embed, neg_train_embed], dim = 0)
    #X_train = torch.cat([pos_embed[pos_train_indices], neg_embed[neg_train_indices]], dim = 0)
    context_train = torch.cat([pos_cell_embed[pos_train_indices], neg_cell_embed[neg_train_indices]], dim = 0) #Added

    groups_train = [pos_prots_group[ind] for ind in pos_train_indices] + [neg_prots_group[ind] for ind in neg_train_indices]
    groups_train_pos = [pos_prots_group[ind] for ind in pos_train_indices]
    groups_train_neg = [neg_prots_group[ind] for ind in neg_train_indices]
    cts_train = [pos_prots_strata[ind] for ind in pos_train_indices] + [neg_prots_strata[ind] for ind in neg_train_indices]
    y_train = np.concatenate([np.ones(len(pos_train_indices)), np.zeros(len(neg_train_indices))])
    
    X_test = dict()
    context_test = dict()
    
    groups_test = dict()
    groups_test_pos = []
    groups_test_neg = []
    for cat in celltype_dict:
        pos_cat_embs = [pos_embed[ind] for ind in pos_test_indices if pos_prots_strata[ind] == cat]
        neg_cat_embs = [neg_embed[ind] for ind in neg_test_indices if neg_prots_strata[ind] == cat]
        pos_cell_embs = [pos_cell_embed[ind] for ind in pos_test_indices if pos_prots_strata[ind] == cat]
        neg_cell_embs = [neg_cell_embed[ind] for ind in neg_test_indices if neg_prots_strata[ind] == cat]
        
        if len(pos_cat_embs) > 0 and len(neg_cat_embs) > 0:
            X_test[cat] = [embedding for embedding_list in [pos_cat_embs, neg_cat_embs] for embedding in embedding_list]
            #X_test[cat] = torch.cat([pos_cat_embs, neg_cat_embs])
            #X_test[cat] = torch.cat([torch.stack(pos_cat_embs), torch.stack(neg_cat_embs)])
            context_test[cat] = torch.cat([torch.stack(pos_cell_embs), torch.stack(neg_cell_embs)]) #Added

            groups_test[cat] = [pos_prots_group[ind] for ind in pos_test_indices if pos_prots_strata[ind] == cat] + [neg_prots_group[ind] for ind in neg_test_indices if neg_prots_strata[ind] == cat]
            groups_test_pos.extend([pos_prots_group[ind] for ind in pos_test_indices if pos_prots_strata[ind] == cat])
            groups_test_neg.extend([neg_prots_group[ind] for ind in neg_test_indices if neg_prots_strata[ind] == cat])
            
            assert len(set(groups_test[cat]).intersection(set(groups_train))) == 0, set(groups_test[cat]).intersection(set(groups_train))

        elif len(pos_cat_embs) == 0 and len(neg_cat_embs) > 0:
            print("Cell type has only negative examples:", cat)
            assert len([pos_prots_group[ind] for ind in pos_test_indices if pos_prots_strata[ind] == cat]) == 0
        elif len(pos_cat_embs) > 0 and len(neg_cat_embs) == 0:
            print("Cell type has only positive examples:", cat)
            X_test[cat] = pos_cat_embs
            #X_test[cat] = torch.stack(pos_cat_embs)
            context_test[cat] = torch.stack(pos_cell_embs)
            assert len([neg_prots_group[ind] for ind in neg_test_indices if neg_prots_strata[ind] == cat]) == 0
            groups_test[cat] = [pos_prots_group[ind] for ind in pos_test_indices if pos_prots_strata[ind] == cat]
            groups_test_pos.extend([pos_prots_group[ind] for ind in pos_test_indices if pos_prots_strata[ind] == cat])
            
            assert len(set(groups_test[cat]).intersection(set(groups_train))) == 0
        else:
            print("Cell type has no positive or negative examples:", cat)

    for k, v in groups_test.items():
        assert len(set(v).intersection(set(groups_train))) == 0, (k, set(v).intersection(set(groups_train)))

    data_split_names_path = data_split_path.split(".json")[0] + "_name.json"
    print(data_split_names_path)
    if not os.path.exists(data_split_names_path): # Generate new splits
        indices_name_dict = {"pos_train_names": list(set(groups_train_pos)),
                             "pos_test_names": list(set(groups_test_pos)),
                             "neg_train_names": list(set(groups_train_neg)),
                             "neg_test_names": list(set(groups_test_neg))}
        for k1, v1 in indices_name_dict.items():
            for k2, v2 in indices_name_dict.items():
                if k1 == k2: continue
                assert len(set(v1).intersection(set(v2))) == 0, (k1, k2)

        with open(data_split_names_path, "w") as outfile:
            json.dump(indices_name_dict, outfile)

    return X_train, X_test, context_train, context_test, y_train, y_test, groups_train, cts_train, groups_test



def main():

    """
    
    Requirements for running this script
        - Json file of positive proteins (dict)     {"<celltype name>": ["<protein name>"]}
        - Json file of negative proteins (dict)     {"<celltype name>": ["<protein name>"]}
        - Json file of raw data (list)              ["<protein name>"]

    Output of this script
        - Json file of data split indices (dict)    {"pos_train_indices": [...], "neg_train_indices": [...], "pos_test_indices": [...], "neg_test_indices": [...]}
        - Json file of data split names (dict)      {"pos_train_names": [...], "neg_train_names": [...], "pos_test_names": [...], "neg_test_names": [...]}
    
    """

    args = read_args()

    # PINNACLE pretrained representations
    embed_path = args.embeddings_dir + args.embed + "_protein_embed.pth"
    labels_path = args.embeddings_dir + args.embed + "_labels_dict.txt"

    # Cell type specific PPI networks
    # celltype_protein_dict = load_PPI_data(args.celltype_ppi)
    data_split_path = args.data_split_path + "_" + args.tt_disease + ".json"

    # Load data
    positive_proteins_prefix = args.positive_proteins_prefix + "_" + args.tt_disease
    negative_proteins_prefix = args.negative_proteins_prefix + "_" + args.tt_disease

    embed, celltype_dict, celltype_protein_dict, positive_proteins, negative_proteins, all_relevant_proteins = load_data(embed_path, labels_path, positive_proteins_prefix, negative_proteins_prefix, args.raw_data_prefix, None)
    for c, v in positive_proteins.items():
        assert len(v) == len(set(v).intersection(set(all_relevant_proteins)))

    # Split data
    process_and_split_data(embed, positive_proteins, negative_proteins, celltype_protein_dict, celltype_dict, data_split_path, random_state=args.random_state, test_size=1-args.train_size-args.val_size)


if __name__ == '__main__':
    main()
