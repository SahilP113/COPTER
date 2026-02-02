## Context-Aware Protein Representations Using Protein Language Models and Optimal Transport 

This repository contains implemention code for COPTER (Contextualized Protein Embeddings via Optimal Transport).

# Overview

Proteins have different functions in different contexts. As a result, representations that take into account a protein’s biological context would allow for a more accurate assessment of its functions and properties. Protein language models (PLMs) generate amino-acid-level (residue-level) embeddings of proteins and are a powerful approach for creating universal protein representations. However, PLMs on their own do not consider context and cannot generate context-specific protein representations. We introduce COPTER, a method that uses optimal transport to pool together a protein's PLM-generated residue-level embeddings using a separate context embedding to create context-aware protein representations. We conceptualize the residue-level embeddings as samples from a probabilistic distribution, and use sliced Wasserstein distances to map these samples against a context-specific reference set, yielding a contextualized protein-level embedding. We evaluate COPTER's performance on three downstream prediction tasks: therapeutic drug target prediction, genetic perturbation response prediction, and TCR-epitope binding prediction. Compared to state-of-the-art baselines, COPTER achieves substantially improved, near-perfect performance in predicting therapeutic targets across cell contexts. It also results in improved performance in predicting responses to genetic perturbations and binding between TCRs and epitopes.

# Run Experiments

Predicting therapeutic targets of rheumatoid arthritis (RA) with ESM-2 8M:
```bash
python train_tt.py --task_name RA_esm2 --tt_disease  EFO_0000685 --plm esm2_8m --embeddings_dir data/therapeutic_target_data/pinnacle_embeds/ --hidden_dim_1 128 --hidden_dim_2 32 --batch_size 32 --num_seeds 1 --num_epoch 150
```

Predicting therapeutic targets of rheumatoid arthritis (RA) with Progen2 Small:
```bash
python train_tt.py --task_name IBD_esm2 --tt_disease EFO_0000685 --plm progen2_small --embeddings_dir data/therapeutic_target_data/pinnacle_embeds/ --hidden_dim_1 256 --hidden_dim_2 64 --batch_size 32 --num_seeds 1 --num_epoch 100
```

Predicting therapeutic targets of inflammatory bowel disease (IBD) with ESM-2 8M:
```bash
python train_tt.py --task_name RA_progen2 --tt_disease  EFO_0003767 --plm esm2_8m --embeddings_dir data/therapeutic_target_data/pinnacle_embeds/ --hidden_dim_1 128 --hidden_dim_2 32 --batch_size 32 --num_seeds 1 --num_epoch 100
```

Predicting therapeutic targets of inflammatory bowel disease (IBD) with Progen2 Small:
```bash
python train_tt.py --task_name IBD_progen2 --tt_disease  EFO_0003767 --plm progen2_small --embeddings_dir data/therapeutic_target_data/pinnacle_embeds/ --hidden_dim_1 256 --hidden_dim_2 64 --batch_size 32 --num_seeds 1 --num_epoch 100
```


# Therapeutic Target Prediction

# TCR-Epitope Binding Prediction

# Genetic Perturbation Response Prediction
