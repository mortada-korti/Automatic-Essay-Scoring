#!/usr/bin/env python
# coding: utf-8

# # Data_preprocessing.py script

# ## Import Required Libraries

# In[1]:


from sklearn.model_selection import train_test_split  # For splitting data into train/val/test
import pandas as pd  # For reading and saving CSV files
import numpy as np  # For fast numerical operations (e.g., arrays, math)
import warnings  # To manage warning messages
import sys, os  # For system paths and file handling


# ## Disable some warnings for cleaner output

# In[2]:


warnings.filterwarnings('ignore')  # Ignore all warnings for simplicity


# ## Setup Project Paths 

# In[3]:


ROOT_DIR = os.path.abspath(os.path.join(os.getcwd(), ".."))  # Get root directory of the project
sys.path.append(ROOT_DIR)  # Add root to system path to allow imports from other scripts

# Import project configuration and helper functions
from config import DATA_DIR
from exec_files.utils import normalize_score


# ## Load the original raw dataset

# In[4]:


dataset = pd.read_csv(f"{ROOT_DIR}/dataset.tsv", delimiter='\t', encoding='ISO-8859-1')

# Select only the required columns from the dataset
df = dataset[["essay_id", "essay_set", "essay", "domain1_score"]]

# Apply score normalization
df = normalize_score(df)


# ## Split into Train, Validation, and Test Sets

# In[5]:


# First split: train (80%) and temp (20%)
train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df["essay_set"], random_state=42)

# Second split: split temp into val (10%) and test (10%)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df["essay_set"], random_state=42)


# ## Save all three sets as tab-separated files

# In[6]:


train_df.to_csv(f"{DATA_DIR}/train_set.tsv", sep="\t", index=False)
val_df.to_csv(f"{DATA_DIR}/val_set.tsv", sep="\t", index=False)
test_df.to_csv(f"{DATA_DIR}/test_set.tsv", sep="\t", index=False)


# ## Check distribution to make sure splits are balanced

# In[7]:


print(df["essay_set"].value_counts(normalize=True))         # original
print(train_df["essay_set"].value_counts(normalize=True))   # training
print(val_df["essay_set"].value_counts(normalize=True))     # validation
print(test_df["essay_set"].value_counts(normalize=True))    # test

