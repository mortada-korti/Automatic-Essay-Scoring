#!/usr/bin/env python
# coding: utf-8

# # Test.py script

# ## Import Required Libraries

# In[1]:


from transformers import DistilBertTokenizerFast, DistilBertModel  # Pretrained model and tokenizer
from torch.utils.data import DataLoader  # PyTorch tool to efficiently load data in batches
import pandas as pd  # For reading and saving CSV files
import warnings  # To manage warning messages
import torch  # Core deep learning library
import sys, os, re  # For system paths and file handling


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
from exec_files.utils import eval_model, encode_dataset, set_seed

# Set random seeds for reproducibility of results
set_seed(42)


# ## Load Test Dataset

# In[4]:


# Read the test dataframe (with tab delimiter) using pandas
test_df = pd.read_csv(f"{DATA_DIR}/test_set.tsv", delimiter="\t", encoding='ISO-8859-1')


# ## Load Tokenizer and Device

# In[5]:


tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")  # Load pre-trained tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available, else CPU


# ## Prepare the Test Dataset for Model Input 

# In[6]:


# This function encodes the text data (tokenizes and converts to tensors)
test_dataset = encode_dataset(test_df, tokenizer)

# Wrap encoded dataset into a DataLoader to allow batch processing
test_loader = DataLoader(test_dataset, batch_size=8)


# ## Evaluate All Trained Models and Store Results

# In[ ]:


# Create an empty list to store results for each tested model
test_results = []

# Define a pattern to extract learning rate, batch size, dropout, epoch, and validation QWK from model filenames
pattern = r"model_lr(.*?)_bs(.*?)_drop(.*?)_epoch(.*?)_qwk(.*?)\.pt"

# Loop over all model checkpoint files in the model directory
for filename in os.listdir(MODEL_DIR):
    # Only consider files that follow naming convention of trained models
    if filename.endswith(".pt") and filename.startswith("model_"):
        match = re.match(pattern, filename)
        if not match:
            continue  # If filename doesn't match expected pattern, skip

        # Extract hyperparameters from filename
        lr, bs, dropout, epoch, val_qwk = match.groups()

        # Load the model checkpoint (saved weights)
        checkpoint_path = os.path.join(MODEL_DIR, filename)
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Re-initialize the base DistilBERT model and the regressor head
        model = DistilBertModel.from_pretrained("distilbert-base-uncased")
        regressor = torch.nn.Sequential(
            torch.nn.Dropout(float(dropout)),  # same dropout rate
            torch.nn.Linear(768, 1)  # regression output (1 value per essay)
        )

        # Load the saved weights into the model and regressor
        model.load_state_dict(checkpoint["bert_state_dict"])
        regressor.load_state_dict(checkpoint["regressor_state_dict"])

        # Move model to the device (CPU or GPU)
        model.to(device)
        regressor.to(device)

        # Evaluate the model on the test set using the evaluation function
        test_qwk = eval_model(test_loader, test_df.copy(), device, model, regressor)

        # Store test result for this model
        test_results.append({
            "learning_rate": float(lr),
            "batch_size": int(bs),
            "dropout": float(dropout),
            "epoch": int(epoch),
            "val_qwk": float(val_qwk),  # QWK on validation set (from training phase)
            "test_qwk": round(test_qwk, 4)  # QWK on unseen test set
        })


# ## Save All Evaluation Results to CSV 

# In[ ]:


# Convert list of results to pandas DataFrame
results_df = pd.DataFrame(test_results) 

# Define path to save test results
results_path = os.path.join(RESULTS_DIR, "test_results.csv")

# Save sorted results by test QWK score (best at top)
results_df.sort_values(by="test_qwk", ascending=False).to_csv(results_path, index=False)

# Print confirmation message
print(f"\nAll models evaluated. Results saved to {results_path}")

