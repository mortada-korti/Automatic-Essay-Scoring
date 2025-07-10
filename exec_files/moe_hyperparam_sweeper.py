#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from transformers import AutoTokenizer, TrainingArguments, Trainer
from sklearn.metrics import cohen_kappa_score 
from datetime import datetime  
from datasets import Dataset  
from pathlib import Path
import pandas as pd
import itertools
import warnings  
import sys, os  
warnings.filterwarnings('ignore') 


# In[ ]:


os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Disable parallel tokenizer threads (prevents unwanted warnings)
os.environ["TORCHINDUCTOR_DISABLE"] = "1"       # Disable TorchInductor (experimental compiler, can cause instability)
os.environ["TORCH_COMPILE_DISABLE"] = "1"       # Disable PyTorch 2.0 compile mode (to avoid compatibility issues)
os.environ["TORCHDYNAMO_DISABLE"] = "1"         # Disable TorchDynamo (used in PyTorch graph capture)


# In[ ]:


ROOT_DIR = os.path.abspath(os.path.join(os.getcwd(), ".."))  # Get the parent directory of the current working directory
sys.path.append(ROOT_DIR)  # Add the root directory to Python's import path so we can import custom modules


# In[ ]:


import torch  

# Import directory paths from config (used to load/save data, models, and results)
from config import DATA_DIR, MODEL_DIR, RESULTS_DIR

# Import custom utility functions from your project
from exec_files.utils import (
    denormalize_score,              # Converts normalized scores back to original scale                
    freeze_bert_layers,             # Freezes all but the top few layers of BERT (partial fine-tuning)
    EssayScoringMoEModel,           # Full essay scoring model using a transformer encoder + MoE head
    generate_train7_test1_splits    # Generates 8 train/test splits for cross-prompt evaluation (no validation)
)


# In[ ]:


PRETRAINED_MODEL = "ModernBERT"  # Identifier for the model used in this experiment
MODEL_NAME = "answerdotai/ModernBERT-base"  # Full Hugging Face model name (used for loading tokenizer & weights)
timestamp = datetime.now().strftime("%Y%m%d_%H%M")  # Generate a unique timestamp (e.g., for saving model files)


# In[ ]:


PRETRAINED_MODEL_DIR = os.path.join(MODEL_DIR, f"{PRETRAINED_MODEL}")  # Create full path to save this model's checkpoints
os.makedirs(PRETRAINED_MODEL_DIR, exist_ok=True)  # Create the directory if it doesn't already exist


# In[ ]:


df = pd.read_csv(f"{DATA_DIR}/dataset.tsv", delimiter="\t", encoding='ISO-8859-1')  # Load the full essay dataset as a DataFrame


# In[ ]:


# Use GPU if available, otherwise fallback to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

# Print how many CUDA-compatible GPUs are available
print(f"CUDA device count: {torch.cuda.device_count()}")

# Print the ID and name of the currently selected GPU device
print(f"Current device: {torch.cuda.current_device()} - {torch.cuda.get_device_name(torch.cuda.current_device())}")


# In[ ]:


def preprocess(example):
    '''
    Tokenizes a single essay and adds its normalized score as the label.
    Used to prepare each sample for training or evaluation.
    '''
    tokens = tokenizer(example["essay"], truncation=True, padding="max_length", max_length=512)
    tokens["labels"] = example["normalized_score"]
    return tokens


# In[ ]:


# Define search space
config_space = {
    "learning_rate": [1e-5, 2e-5, 3e-5, 4e-5, 5e-5],
    "batch_size": [8, 16, 32],
    "epochs": [5, 7, 10],
    "dropout": [0.2],
    "num_experts": [7],
    "aux_loss_weight": [0.5]
}

# Build all combinations
config_list = list(itertools.product(*config_space.values()))
config_keys = list(config_space.keys())


# In[ ]:


# Load previous results if the file exists
sweep_results_path = Path(f"{RESULTS_DIR}/hyperparam_sweep_results.csv")
if sweep_results_path.exists():
    existing_df = pd.read_csv(sweep_results_path)
    existing_configs = existing_df[config_keys].to_dict("records")
else:
    existing_df = None
    existing_configs = []


# In[ ]:


# Generate a list of 8 cross-prompt splits (train on 7 essay sets, test on 1)
# No validation set is used in this setup
splits = generate_train7_test1_splits()

# List to collect performance results (QWKs + config info)
results = []

# Loop through each hyperparameter configuration
for config_id, values in enumerate(config_list):
    config = dict(zip(config_keys, values))  # Create a dictionary from keys and current values

    # Skip this config if results already exist (for resume-safe training)
    if config in existing_configs:
        print(f"*** Skipping Config {config_id+1}/{len(config_list)} — already completed ***")
        continue

    # Print header for the current config run
    print("\n", "-"*135)
    print(f"\nRunning Config {config_id+1}/{len(config_list)}: {config}")
    print("-"*135, "\n")
    
    test_qwks = []  # Store test QWK for each split in this config

    # Loop through each train/test split
    for split in splits:
        # Filter training and test data based on current essay sets
        train_df = df[df["essay_set"].isin(split["train"])].copy()
        test_df = df[df["essay_set"] == split["test"]].copy()

        # Load tokenizer and initialize the MoE-based essay scoring model
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = EssayScoringMoEModel(
            MODEL_NAME,
            num_experts=config["num_experts"],   # set number of experts from config
            dropout=config["dropout"]            # set dropout rate from config
        )

        # Freeze all but top 4 layers of the transformer
        freeze_bert_layers(model.encoder, num_unfrozen=4)

        # Set the auxiliary loss weight dynamically inside the model config
        model.encoder.config.aux_loss_weight = config["aux_loss_weight"]

        # Convert dataframes to Hugging Face Dataset objects
        train_dataset = Dataset.from_pandas(train_df)
        test_dataset = Dataset.from_pandas(test_df)

        # Apply tokenization and formatting
        train_dataset = train_dataset.map(preprocess)
        test_dataset = test_dataset.map(preprocess)

        # Define training configuration for this run
        training_args = TrainingArguments(
            per_device_train_batch_size=config["batch_size"],  # set batch size
            per_device_eval_batch_size=config["batch_size"],
            learning_rate=config["learning_rate"],
            num_train_epochs=config["epochs"],
            eval_strategy="no",       # no validation during training
            save_strategy="no",       # don't save intermediate checkpoints
            logging_strategy="no",    # don't log intermediate metrics
            report_to="none",         # disable integration with W&B, TensorBoard, etc.
            weight_decay=0.01,        # regularization
            warmup_ratio=0.1          # learning rate warmup for stability
        )

        # Use Hugging Face Trainer (custom model + setup)
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=tokenizer
        )

        # Train the model on current train set
        trainer.train()

        # Make predictions on the test set
        predictions = trainer.predict(test_dataset)
        preds = predictions.predictions
        labels = predictions.label_ids

        # Convert normalized predictions/labels back to original scale
        preds = denormalize_score(preds, test_df)
        labels = denormalize_score(labels, test_df)

        # Access gate weights π from the model after forward pass
        gate_weights = model.last_gate_weights.numpy()  # shape: (batch_size, num_experts)
        avg_gate_weights = gate_weights.mean(axis=0)    # average π values across samples

        # Compute QWK score on the test set
        qwk = cohen_kappa_score(labels, preds, weights="quadratic")
        test_qwks.append(qwk)

        print("\n", "-"*50)
        print(f"Config {config_id+1}/{len(config_list)} -> prompt_{split['test']} -> Test QWK: {qwk:.4f}")
        print("-"*50, "\n")

    # Compute average QWK across all test splits for this config
    avg_qwk = sum(test_qwks) / len(test_qwks)

    # Create a full result row with config, per-prompt QWKs, and expert π weights
    result_row = {
        **config,  # include learning_rate, dropout, etc.
        **{f"prompt_{i+1}": q for i, q in enumerate(test_qwks)},  # test QWKs
        "avg_qwk": avg_qwk,
        **{f"pi_expert_{i}": avg_gate_weights[i] for i in range(len(avg_gate_weights))}  # average gate trust
    }

    results.append(result_row)  # Add this result to the full list

    # Save results to CSV after each config (to prevent data loss on interruption)
    pd.DataFrame(results).to_csv(f"{RESULTS_DIR}/hyperparam_sweep_results.csv", index=False)

