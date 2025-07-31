#!/usr/bin/env python
# coding: utf-8

# # DeBERTa_trainer.py

# ### Import required libraries

# In[ ]:


from transformers import TrainingArguments, DebertaTokenizer, DebertaConfig, Trainer
from sklearn.metrics import cohen_kappa_score 
from datasets import Dataset  
from pathlib import Path
import pandas as pd
import itertools
import warnings  
import sys, os  

# Hide warnings to keep notebook output clean
warnings.filterwarnings('ignore') 


# ### Environment Flags to Ensure Stable Execution

# In[ ]:


os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Prevents parallel tokenizer threads — avoids potential race conditions or console spam
os.environ["TORCHINDUCTOR_DISABLE"] = "1"       # Disables TorchInductor (experimental compiler backend) to ensure compatibility
os.environ["TORCH_COMPILE_DISABLE"] = "1"       # Disables PyTorch 2.0's torch.compile functionality (can cause issues in custom models)
os.environ["TORCHDYNAMO_DISABLE"] = "1"         # Turns off TorchDynamo, another dynamic optimization engine in PyTorch 


# ### Set Project Root Directory and Add to Python Path

# In[ ]:


ROOT_DIR = os.path.abspath(os.path.join(os.getcwd(), '..', '..'))  # Define the root directory two levels above the current working directory
sys.path.append(ROOT_DIR)   # Add the root directory to Python's module search path so that custom modules can be imported


# ### Import Core Project Modules and Model Components

# In[ ]:


import torch  

# Project-specific configuration: paths for data and output storage
from config import DATA_DIR, RESULTS_DIR

# Import MoE and DeBERTa model components from custom DeBERTa_utils script
from scripts.DeBERTa.DeBERTa_utils import (
    freeze_deberta_layers,
    DebertaLayerWithMoE,
    MoeDebertaScorer,
    MoeDebertaModel,
    MoEFeedForward,
)

# Import utility functions for data processing and scoring
from scripts.utils import generate_train7_test1_splits, denormalize_score, preprocess


# ### Load Essay Dataset from TSV File

# In[ ]:


# Load the essay dataset as a pandas DataFrame.
df = pd.read_csv(f"{DATA_DIR}/dataset.tsv", delimiter="\t", encoding='ISO-8859-1') 


# ### Set Device for Computation (GPU if Available)

# In[ ]:


# Use GPU (CUDA) if available; otherwise fallback to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  

# Print how many CUDA-enabled GPUs are detected
print(f"CUDA device count: {torch.cuda.device_count()}")

# Show the index and name of the active GPU (if one is available)
print(f"Current device: {torch.cuda.current_device()} - {torch.cuda.get_device_name(torch.cuda.current_device())}")


# ### Define Hyperparameter Search Space

# In[ ]:


config_space = {
    "learning_rate": [5e-5],           # Learning rates to try
    "batch_size": [8],                 # Batch size (fixed)
    "epochs": [15],                    # Number of training epochs
    "dropout": [0.2],                  # Dropout rate for regularization
    "num_experts": [7],                # Number of experts in each MoE layer
    "aux_loss_weight": [0.5],          # Weight for auxiliary loss (entropy regularization)
    "unfrozen_layers": [2],            # How many top DeBERTa layers to fine-tune
    "top_k": [2]                       # Number of experts selected per token
}

# Generate all possible combinations of hyperparameters
config_list = list(itertools.product(*config_space.values()))

# Store the corresponding keys to map each config tuple
config_keys = list(config_space.keys())


# ### Load Existing Results (If Any) to Avoid Redundant Experiments

# In[ ]:


results_path = Path(f"{RESULTS_DIR}/DeBERTa/DeBERTa_results.csv")  # Path to CSV where previous results are saved

if results_path.exists():
    existing_df = pd.read_csv(results_path)  # Load previously saved results
    existing_configs = existing_df[config_keys].to_dict("records")  # Extract existing configurations to check for duplicates
else:
    existing_df = None                     # No existing results file found
    existing_configs = []                  # Start fresh with an empty config list


# ### Run Cross-Prompt MoE-DeBERTa Experiments Over All Configurations

# In[ ]:


splits = generate_train7_test1_splits()  # Generate 8-fold cross-prompt train/test splits
results = []                             # Store results for each configuration

tokenizer = DebertaTokenizer.from_pretrained("microsoft/deberta-base")   # Load BERT tokenizer

for config_id, values in enumerate(config_list):
    config = dict(zip(config_keys, values))  # Map hyperparameter values to names

    # Skip this config if results already exist (for resume-safe training)
    if config in existing_configs:
        print(f"*** Skipping Config {config_id+1}/{len(config_list)} — already completed ***")
        continue

    print("\n", "-"*135)
    print(f"\nRunning Config {config_id+1}/{len(config_list)}: {config}")
    print("-"*135, "\n")
    
    test_qwks = []  # Collect QWK scores from each prompt fold

    for split in splits:
        # Split data by prompt
        train_df = df[df["essay_set"].isin(split["train"])].copy()
        test_df = df[df["essay_set"] == split["test"]].copy()
        
        # Convert to Hugging Face Datasets
        train_dataset = Dataset.from_pandas(train_df)
        test_dataset = Dataset.from_pandas(test_df)
        
        # Apply preprocessing (tokenization + handcrafted features)
        train_dataset = train_dataset.map(lambda example: preprocess(example, tokenizer))
        test_dataset = test_dataset.map(lambda example: preprocess(example, tokenizer))

        n_handcrafted_features = len(train_dataset[0]["features"])  # Dimensionality of external features

        # Load DeBERTa config and initialize MoeDeberta model with MoE layers
        deberta_config = DebertaConfig.from_pretrained("microsoft/deberta-base")
        moe_model = MoeDebertaModel(deberta_config, num_experts=config["num_experts"], top_k=config["top_k"])
        model = MoeDebertaScorer(base_model=moe_model, dropout=config["dropout"], feature_dim=n_handcrafted_features)
        model.to(device)

        # Freeze lower layers of DeBERTa
        freeze_deberta_layers(model.encoder, num_unfrozen=config["unfrozen_layers"])

        # Assign auxiliary loss weight for entropy-based expert regularization
        model.encoder.config.aux_loss_weight = config["aux_loss_weight"]
        
        # Define training arguments
        training_args = TrainingArguments(
            per_device_train_batch_size=config["batch_size"],  
            per_device_eval_batch_size=config["batch_size"],
            learning_rate=config["learning_rate"],
            num_train_epochs=config["epochs"],
            eval_strategy="no",       # no validation during training
            save_strategy="no",       # don't save intermediate checkpoints
            logging_strategy="no",    # don't log intermediate metrics
            report_to="none",         # disable integration with W&B, TensorBoard, etc.
            weight_decay=0.01,        # regularization
            warmup_ratio=0.1,         # learning rate warmup for stability
        )

        # Initialize Trainer with model and dataset
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
        )

         # Train the model
        trainer.train()

        # Run prediction on test set
        predictions = trainer.predict(test_dataset)
        preds = predictions.predictions[0]
        labels = predictions.label_ids

        # Convert predictions and labels back to original scoring scale
        preds = denormalize_score(preds, test_df)
        labels = denormalize_score(labels, test_df)

        # Get average expert routing weights
        gate_weights = model.last_gate_weights.numpy()
        avg_gate_weights = gate_weights.mean(axis=0)

        # Evaluate with Quadratic Weighted Kappa
        qwk = cohen_kappa_score(labels, preds, weights="quadratic")
        test_qwks.append(qwk)

        print("\n", "-"*50)
        print(f"Config {config_id+1}/{len(config_list)} -> prompt_{split['test']} -> Test QWK: {qwk:.4f}")
        print("-"*50, "\n")

    # Average QWK across all prompts
    avg_qwk = sum(test_qwks) / len(test_qwks)

    # Prepare result row with metrics and expert routing stats
    result_row = {
        **config,
        **{f"prompt_{i+1}": q for i, q in enumerate(test_qwks)},
        "avg_qwk": avg_qwk,
        **{f"pi_expert_{i}": avg_gate_weights[i] for i in range(len(avg_gate_weights))}
    }

    results.append(result_row)  # Save result to memory

    # Save running results to disk after each config
    pd.DataFrame(results).to_csv(f"{RESULTS_DIR}/DeBERTa/DeBERTa_results.csv", index=False)

