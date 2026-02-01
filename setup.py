# Set up model environment, parameters, and data

import os
import pandas as pd
import argparse
import json
import torch

from read_data import read_labels_from_evidence


def create_parser():
    parser = argparse.ArgumentParser()

    #Graph data for training PINNACLE
    parser.add_argument("--G_f", type=str, default="../therapeutic_target_data/networks/global_ppi_edgelist.txt/", help="Directory to global reference PPI network")
    parser.add_argument("--ppi_dir", type=str, default="../therapeutic_target_data/networks/ppi_edgelists/", help="Directory to PPI layers")
    parser.add_argument("--mg_f", type=str, default="../therapeutic_target_data/networks/mg_edgelist.txt", help="Directory to metagraph")
 

    # MLP model hyperparameters
    parser.add_argument("--hidden_dim_1", type=int, default=64, help="1st hidden dim size")  #Change to 320+64=384 if doing average pooling
    parser.add_argument("--hidden_dim_2", type=int, default=32, help="2nd hidden dim size, discard if 0") 
    parser.add_argument("--hidden_dim_3", type=int, default=0, help="3rd hidden dim size, discard if 0") 
    parser.add_argument("--dropout", type=float, default=0, help="dropout rate")
    parser.add_argument("--norm", type=str, default=None, help="normalization layer")
    parser.add_argument("--actn", type=str, default="relu", help="activation type")
    parser.add_argument("--order", type=str, default="nd", help="order of normalization and dropout")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--wd", type=float, default=1e-4, help="weight decay")
    parser.add_argument("--num_epoch", type=int, default=1, help="epoch num")
    parser.add_argument("--batch_size", type=int, help="batch size")
    
    #Hyperparameters for GNN (therapeutic target prediction task)
    """
    parser.add_argument("--feat_mat", type=int, default=2048, help="Random Gaussian vectors of shape (1 x 2048)")
    parser.add_argument("--output", type=int, default=8, help="Output size")
    parser.add_argument("--hidden", type=int, default=16, help="Output size")
    parser.add_argument("--lr_GNN", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--wd_GNN", type=float, default=5e-4, help="Weight decay")
    parser.add_argument("--dropout_GNN", type=float, default=0.5, help="Dropout")
    parser.add_argument("--n_heads", type=int, default=8, help="Number of heads")
    parser.add_argument("--theta", type=float, default=0.1, help="Theta (for PPI loss)")
    parser.add_argument("--lmbda", type=float, default=0.01, help="Lambda (for center loss function)")
    parser.add_argument("--lr_cent", type=float, default=0.01, help="Learning rate for center loss")
    parser.add_argument("--batch_size_GNN", type=int, default=64, help="Batch size")
    parser.add_argument("--norm_GNN", type=str, default=None, help="Type of normalization layer to use in up-pooling")
    parser.add_argument("--pc_att_channels", type=int, default=8, help="Type of normalization layer to use in up-pooling")
    """
    #For all tasks
    parser.add_argument("--task_name", type=str)  #For all tasks

    #PLM specification
    parser.add_argument("--plm", type=str, default="esm2_8m") #other options: esm2_650m, progen2_small, progen2_large

    #Data for therapeutic target task
    parser.add_argument("--tt_disease", type=str, default="EFO_0000685")  #Specific to therapeutic target task, EFO_0003767 for IBD tasks, EFO_0000685 for RA task
    parser.add_argument("--data_split_path", type=str, default="../therapeutic_target_data/therapeutic_target_task/data_split") #Change to EFO_0003767 for IBD task  #Use EFO_0000685 for RA task   #If using the 5 test cells, ../raw_data/therapeutic_target_task/data_split_EFO_0000685_test 
    parser.add_argument('--positive_proteins_prefix', type=str, default="../therapeutic_target_data/therapeutic_target_task/positive_proteins") #../data/therapeutic_target_task/positive_proteins
    parser.add_argument('--negative_proteins_prefix', type=str, default="../therapeutic_target_data/therapeutic_target_task/negative_proteins") #../data/therapeutic_target_task/negative_proteins
    
    #TCR epitope task type
    parser.add_argument("--tcr_epitope_task", type=str, default="RN")  #RN for RN task, NA for NA task

    #Genetic perturbation task type
    parser.add_arugment("--perturb_task", type=str, default="replogle_rpe1")  #replogle_rpe1 or replogle_k562

    #COPTER representations
    parser.add_argument("--embeddings_dir", type=str)
    parser.add_argument("--embed", type=str, default="pinnacle")
    parser.add_argument("--embed2", type=str, default="pooling")
    #Used tmp_evaluation_results for original PINNACLE code with 1 epoch
    #tmp_evaluation_results2 with combined pooling and PINNACLE for 1 epoch
    #tmp_evaluation_results3 with combined pooling and PINNACLE for 50 epochs

    # Output directories
    parser.add_argument("--metrics_output_dir", type=str, default="./tmp_evaluation_results") #"./tmp_evaluation_results_norman_8M/" #Changed from tmp_evaluation_results
    parser.add_argument("--models_output_dir", type=str, default="./tmp_model_outputs") #"./tmp_model_outputs_norman_8M/" #Changed from tmp_model_outputs
    parser.add_argument("--n_seeds", type=int, default=1)  #Only for therapeutic target and genetic perturbation tasks
    parser.add_argument("--random_state", type=int, default=1)
    parser.add_argument("--random", action="store_true", help="random runs without fixed seeds")
    parser.add_argument("--overwrite", action="store_true", help="whether to overwrite the label data or not")
    parser.add_argument("--train_size", type=float, default=0.6)
    parser.add_argument("--val_size", type=float, default=0.2)
    parser.add_argument("--weigh_sample", action="store_true", help="whether to weigh samples or not")  # default = False
    parser.add_argument("--weigh_loss", action="store_true", help="whether to weigh losses or not")  # default = False

    
    
    args = parser.parse_args()
    return args


def get_hparams(args):
    """
    hparams = {
               "lr": args.lr, 
               "wd": args.wd, 
               "hidden_dim_1": args.hidden_dim_1, 
               "hidden_dim_2": args.hidden_dim_2, 
               "hidden_dim_3": args.hidden_dim_3, 
               "dropout": args.dropout, 
               "actn": args.actn, 
               "order": args.order, 
               "norm": args.norm,
               "task_name": args.task_name,
            
               'pc_att_channels': args.pc_att_channels,
               'feat_mat': args.feat_mat, 
               'output': args.output,
               'hidden': args.hidden,
               'lr_GNN': args.lr_GNN, 
               'wd_GNN': args.wd_GNN,
               'dropout_GNN': args.dropout_GNN,
               'gradclip': 1.0,
               'n_heads': args.n_heads,
               'lambda': args.lmbda,
               'theta': args.theta,
               'lr_cent': args.lr_cent,
               'loss_type': "BCE",
               'plot': args.plot,
              }
    """
    hparams = {
               "lr": args.lr, 
               "wd": args.wd, 
               "hidden_dim_1": args.hidden_dim_1, 
               "hidden_dim_2": args.hidden_dim_2, 
               "hidden_dim_3": args.hidden_dim_3, 
               "dropout": args.dropout, 
               "actn": args.actn, 
               "order": args.order, 
               "norm": args.norm,
               "task_name": args.task_name,
              }
    return hparams


def setup_paths(args):
    random_state = args.random_state if args.random_state >= 0 else None
    if random_state == None:
        models_output_dir = args.models_output_dir + args.embed + "/"
        metrics_output_dir = args.metrics_output_dir + args.embed + "/"
    else:
        models_output_dir = args.models_output_dir + args.embed + ("_seed=%s" % str(random_state)) + "/"
        metrics_output_dir = args.metrics_output_dir + args.embed + ("_seed=%s" % str(random_state)) + "/"
    if not os.path.exists(models_output_dir): os.makedirs(models_output_dir)
    if not os.path.exists(metrics_output_dir): os.makedirs(metrics_output_dir)
    
    embed_path = args.embeddings_dir + args.embed + "_protein_embed.pth"
    labels_path = args.embeddings_dir + args.embed + "_labels_dict.txt"    #Using pinnacle_labels_dict?
    return models_output_dir, metrics_output_dir, random_state, embed_path, labels_path
