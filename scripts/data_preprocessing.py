#!/usr/bin/env python
# coding: utf-8

# # Data_preprocessing.py script

# ### Import Required Libraries

# In[ ]:


import pandas as pd  
import warnings 
import sys, os 


# ### Disable some warnings for cleaner output

# In[ ]:


warnings.filterwarnings('ignore') 


# ### Setup Project Paths 

# In[ ]:


ROOT_DIR = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))
print(ROOT_DIR)
sys.path.append(ROOT_DIR)  

from config import DATA_DIR
from utils import (
    add_features,
    normalize_score,
    extract_length_features,
    extract_readability_features,
    extract_text_complexity_features,
    extract_text_variation_features,
    extract_sentiment_features
)


# ### Load the original raw dataset

# In[ ]:


dataset = pd.read_csv(f"{ROOT_DIR}/dataset.tsv", delimiter='\t', encoding='ISO-8859-1')
df = dataset[["essay_id", "essay_set", "essay", "domain1_score"]]


# In[ ]:


df = normalize_score(df)
df = add_features(df, extract_length_features, "len_")
df = add_features(df, extract_readability_features, "read_")
df = add_features(df, extract_text_complexity_features, "comp_")
df = add_features(df, extract_text_variation_features, "var_")
df = add_features(df, extract_sentiment_features, "sent_")


# In[ ]:


df.to_csv(f"{DATA_DIR}/dataset.tsv", sep="\t", index=False)

