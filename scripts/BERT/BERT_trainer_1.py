#!/usr/bin/env python
# coding: utf-8

# # BERT_trainer_1.py

# ### Import Training Utilities, Metrics, and Setup Tools

# In[ ]:


from transformers import TrainingArguments, BertTokenizer, BertConfig, Trainer
from sklearn.metrics import cohen_kappa_score 
from datasets import Dataset  
from pathlib import Path 
import pandas as pd  
import numpy as np
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

# Import MoE and BERT model components from custom BERT_utils script
from scripts.BERT.BERT_utils_1 import (
    map_essay_set_to_expert,
    freeze_bert_layers,
    BertLayerWithMoE,
    add_expert_mask,
    MoEFeedForward,
    data_collator,
    MoeBERTScorer,
    MoeBERTModel,
    preprocess
)

# Import utility functions for data processing and scoring
from scripts.utils import denormalize_score , generate_train7_test1_splits     


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
    "learning_rate": [5e-5],    # Learning rates to try
    "batch_size": [8],          # Per-device batch size
    "epochs": [7, 20],          # Training epochs
    "dropout": [0.2],           # Dropout rate
    "num_experts": [7],         # Experts per MoE layer
    "aux_loss_weight": [0.0],   # Weight for auxiliary lossd
    "unfrozen_layers": [6],     # Unfrozen BERT layers
    "top_k": [7],               # Experts selected per token
    "grad_accum_steps": [0]     # gradient accumulation steps
}
# Generate all possible combinations of hyperparameters
config_list = list(itertools.product(*config_space.values()))

# Store the corresponding keys to map each config tuple
config_keys = list(config_space.keys())


# ### Load Existing Results (If Any) to Avoid Redundant Experiments

# In[ ]:


results_path = Path(f"{RESULTS_DIR}/BERT/BERT_results_55.csv")  # Path to CSV where previous results are saved

if results_path.exists():
    existing_df = pd.read_csv(results_path)  # Load previously saved results
    existing_configs = existing_df[config_keys].to_dict("records")  # Extract existing configurations to check for duplicates
else:
    existing_df = None                     # No existing results file found
    existing_configs = []                  # Start fresh with an empty config list


# ### Run Cross-Prompt MoE-BERT Experiments Over All Configurations

# In[ ]:


# --- Main training + evaluation loop ---
# Purpose:
# Runs multiple experiments across different hyperparameter configs and cross-prompt splits.
# - Prepares data for each split
# - Builds and trains a MoeBERTScorer model
# - Evaluates on the test set
# - Logs performance and gating stats to a CSV

splits = generate_train7_test1_splits()  # Create 8-fold cross-prompt train/test splits
results = []

# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Loop over all hyperparameter configurations
for config_id, values in enumerate(config_list):
    config = dict(zip(config_keys, values))  # Map config values to their names

    # Skip configs that have already been run
    if config in existing_configs:
        print(f"*** Skipping Config {config_id+1}/{len(config_list)} — already completed ***")
        continue

    # Print config header
    print("\n", "-"*135)
    print(f"\nRunning Config {config_id+1}/{len(config_list)}: {config}")
    print("-"*135, "\n")

    test_qwks = []  # Store QWK results for each test prompt

    # Loop over each train/test split
    for split in splits:
        # Separate training and test data by essay_set
        train_df = df[df["essay_set"].isin(split["train"])].copy()
        test_df  = df[df["essay_set"] == split["test"]].copy()

        # Convert DataFrames to Hugging Face Dataset objects
        train_dataset = Dataset.from_pandas(train_df)
        test_dataset  = Dataset.from_pandas(test_df)

        # Tokenize text and extract handcrafted features
        train_dataset = train_dataset.map(lambda ex: preprocess(ex, tokenizer))
        test_dataset  = test_dataset.map(lambda ex: preprocess(ex, tokenizer))

        # Create a fixed mapping from essay_set → expert index (train only)
        expert_map = map_essay_set_to_expert(split["train"])
        print("Expert assignment per essay_set:")
        for es, expert_id in expert_map.items():
            print(f"  Essay Set {es} → Expert {expert_id}")

        # Add expert_mask to training set (used for gating supervision)
        train_dataset = train_dataset.map(lambda ex: add_expert_mask(ex, expert_map))
        # Note: test_dataset does NOT get expert_mask (soft gating only on unseen prompts)

        # Number of handcrafted features per essay
        n_handcrafted_features = len(train_dataset[0]["features"])

        # Configure and create the MoE BERT model
        bert_config = BertConfig.from_pretrained("bert-base-uncased")
        moe_model = MoeBERTModel(
            bert_config,
            num_experts=config["num_experts"],
            top_k=config["top_k"],
            pretrained_name_or_path="bert-base-uncased"
        )
        model = MoeBERTScorer(
            base_model=moe_model,
            dropout=config["dropout"],
            feature_dim=n_handcrafted_features
        ).to(device)

        # Freeze lower BERT layers according to config
        freeze_bert_layers(model.encoder, num_unfrozen=config["unfrozen_layers"])
        model.encoder.config.aux_loss_weight = config["aux_loss_weight"]

        # Hugging Face training arguments
        training_args = TrainingArguments(
            per_device_train_batch_size=config["batch_size"],
            per_device_eval_batch_size=config["batch_size"],
            learning_rate=config["learning_rate"],
            num_train_epochs=config["epochs"],
            output_dir="./checkpoints",
            label_names=["labels"],
            logging_strategy="no",
            eval_strategy="no",
            save_strategy="no",
            report_to="none",
            max_grad_norm=1.0,
            weight_decay=0.01,
            warmup_ratio=0.1
            #gradient_accumulation_steps=config["grad_accum_steps"], 
        )

        # Trainer handles training loop, batching, and optimization
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )

        trainer.train()  # Train the model

        # ---- Evaluation on test set ----
        test_predictions = trainer.predict(test_dataset)

        # Extract predictions array
        raw_preds = test_predictions.predictions
        if isinstance(raw_preds, (list, tuple)):
            raw_preds = raw_preds[0]
            
        raw_preds = np.squeeze(raw_preds)
        raw_labels = np.squeeze(test_predictions.label_ids)

        # Convert scores back to original scale
        test_preds  = denormalize_score(raw_preds,  test_df)
        test_labels = denormalize_score(raw_labels, test_df)

        # ---- Gating statistics ----
        gate_weights = model.last_gate_weights.numpy()  # avg gate weights per sequence
        avg_gate_weights = gate_weights.mean(axis=0)    # mean across batch
        prob_dist = avg_gate_weights / (avg_gate_weights.sum() + 1e-12)
        entropy = -(prob_dist * np.log(prob_dist + 1e-12)).sum()
        top_expert = int(np.argmax(avg_gate_weights))

        # Quadratic Weighted Kappa score for performance
        qwk_test = cohen_kappa_score(test_labels, test_preds, weights="quadratic")
        test_qwks.append(qwk_test)

        print("\n", "-"*50)
        print(f"Config {config_id+1}/{len(config_list)} -> prompt_{split['test']} -> Test QWK: {qwk_test:.4f}")
        print("-"*50, "\n")

        # Save this run’s results
        result_row = {
            "train_sets": split["train"],
            "test_set": split["test"],
            "prompt_test_qwk": qwk_test,
            "gate_entropy": entropy,
            "top_expert": top_expert,
            **{f"pi_expert_{i}": float(avg_gate_weights[i]) for i in range(len(avg_gate_weights))},
            **config
        }

        results_df = pd.DataFrame([result_row])
        results_csv_path = f"{RESULTS_DIR}/BERT/BERT_results_55.csv"
        if os.path.exists(results_csv_path):
            results_df.to_csv(results_csv_path, mode="a", header=False, index=False)
        else:
            results_df.to_csv(results_csv_path, mode="w", header=True, index=False)

        # Free GPU memory
        del trainer, model
        torch.cuda.empty_cache()

