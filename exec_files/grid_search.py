#!/usr/bin/env python
# coding: utf-8

# # Grid_search.py script

# ## Import Required Libraries

# In[1]:


from transformers import DistilBertTokenizerFast, DistilBertModel  # Pretrained model and tokenizer
from torch.utils.data import DataLoader  # PyTorch tool to efficiently load data in batches
from itertools import product  # Helps try every combination of parameters
import pandas as pd  # For reading and saving CSV files
import warnings  # To manage warning messages
import torch  # Core deep learning library
import sys, os  # For system paths and file handling


# ## Disable some warnings for cleaner output

# In[2]:


os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Avoid tokenizer parallelism warnings
warnings.filterwarnings('ignore')  # Ignore all warnings for simplicity


# ## Setup Project Paths 

# In[3]:


ROOT_DIR = os.path.abspath(os.path.join(os.getcwd(), ".."))  # Get root directory of the project
sys.path.append(ROOT_DIR)  # Add root to system path to allow imports from other scripts

# Import project configuration and helper functions
from config import DATA_DIR, MODEL_DIR, RESULTS_DIR
from exec_files.utils import train_model, encode_dataset, set_seed

# Set random seeds for reproducibility of results
set_seed(42)


# ## Define run_grid_search function

# In[4]:


def run_grid_search():
    # === Define Hyperparameter Search Space ===
    learning_rates = [1e-5, 2e-5, 3e-5]  # How fast the model learns
    batch_sizes = [8, 16, 32]  # How many essays to process at once
    dropout_rates = [0.1, 0.2, 0.3]  # To prevent overfitting
    csv_path = f"{RESULTS_DIR}/grid_search_results.csv"  # Where results are saved

    # Load Tokenizer and Dataframes 
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")  # Load pre-trained tokenizer
    # Read the train dataframe (with tab delimiter) using pandas
    train_df = pd.read_csv(f"{DATA_DIR}/train_set.tsv", delimiter="\t", encoding='ISO-8859-1')
    # Read the validation dataframe (with tab delimiter) using pandas
    val_df = pd.read_csv(f"{DATA_DIR}/val_set.tsv", delimiter="\t", encoding='ISO-8859-1')

    # Track Best Model So Far 
    best_qwk = -1
    best_model_state = None
    best_regressor_state = None
    best_epoch = 0
    best_config = None
    tried_configs = set()  # Helps avoid retraining same setup

    # Load Past Results to Avoid Repeating 
    if os.path.exists(csv_path):
        prev_df = pd.read_csv(csv_path)
        tried_configs = set(tuple(row) for row in prev_df[["learning_rate", "batch_size", "dropout"]].values)

        # Get best config from past runs (if any)
        best_qwk = prev_df["qwk"].max()
        best_row = prev_df.loc[prev_df["qwk"].idxmax()]
        best_config = {"lr": best_row["learning_rate"], "bs": best_row["batch_size"], "dropout": best_row["dropout"]}
        best_epoch = int(best_row["epoch"])

    # Ensure Model Directory Exists 
    os.makedirs(f"{MODEL_DIR}", exist_ok=True)
    
    for lr, batch_size, dropout in product(learning_rates, batch_sizes, dropout_rates):
        current_config = (lr, batch_size, dropout)
        if current_config in tried_configs:
            print(f"\nSkipping already tried: lr={lr}, bs={batch_size}, dropout={dropout}")
            continue  # Skip if we've already tried this combo before
    
        print(f"\nTraining with: learning_rate={lr}, batch_size={batch_size}, dropout={dropout}")

        # This function encodes the text data (tokenizes and converts to tensors)
        train_dataset = encode_dataset(train_df, tokenizer, label_column="normalized_score")
        val_dataset = encode_dataset(val_df, tokenizer, label_column="normalized_score")

        # Wrap encoded dataset into a DataLoader to allow batch processing
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # Initialize Model and Regression Head 
        distil_bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        regressor = torch.nn.Sequential(torch.nn.Dropout(dropout), torch.nn.Linear(768, 1))

        # Move to GPU (if available) 
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        distil_bert.to(device)
        regressor.to(device)

        # Train Model on This Config 
        qwk, model_state, regressor_state, epoch = train_model(
            train_loader, val_loader, val_df.copy(), device,
            distil_bert, regressor, num_epochs=10, learning_rate=lr
        )

        # Save Training Result to CSV 
        result_row = {
            "learning_rate": lr,
            "batch_size": batch_size,
            "dropout": dropout,
            "epoch": epoch,
            "qwk": qwk
        }
        pd.DataFrame([result_row]).to_csv(csv_path, mode="a", header=not os.path.exists(csv_path), index=False)

        # Save Model Checkpoint 
        model_name = f"model_lr{lr}_bs{batch_size}_drop{dropout}_epoch{epoch}_qwk{qwk:.4f}.pt"
        torch.save({
            "bert_state_dict": model_state,
            "regressor_state_dict": regressor_state,
            "epoch": epoch,
            "qwk": qwk,
            "config": {
                "learning_rate": lr,
                "batch_size": batch_size,
                "dropout": dropout
            }
        }, os.path.join(f"{MODEL_DIR}", model_name))

        print(f"\nFinished training, model saved as: {model_name}.")


# ## Run the Search When File is Executed

# In[5]:


# Run the grid search only if this file is executed as the main script
if __name__ == "__main__":
    run_grid_search()

