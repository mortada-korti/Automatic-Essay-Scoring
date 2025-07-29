#!/usr/bin/env python
# coding: utf-8

# # Utils.py 

# ### Imports and Language Model Initialization

# In[ ]:


from sklearn.preprocessing import MinMaxScaler  
from collections import Counter  
from textblob import TextBlob  
import pandas as pd  
import numpy as np
import textstat  
import spacy  
import math  
import re  

# Load a lightweight English NLP model from spaCy
nlp = spacy.load("en_core_web_sm")


# ### Normalize Essay Scores by Prompt

# In[ ]:


def normalize_score(df):
    """
    Normalize essay scores within each essay set to a 0–1 range.

    Parameters:
        df: A dataframe containing at least 'essay_set' and 'domain1_score' columns.

    Returns:
        pd.DataFrame: The original dataframe with added columns for min, max, and normalized scores.
    """
    df = df.copy()  # Avoid modifying the original dataframe

    # Calculate the minimum score for each essay set (prompt)
    df["score_min"] = df.groupby("essay_set")["domain1_score"].transform("min")

    # Calculate the maximum score for each essay set
    df["score_max"] = df.groupby("essay_set")["domain1_score"].transform("max")

    # Normalize the score to a range of 0–1 based on the min and max for the same essay set
    df["normalized_score"] = (df["domain1_score"] - df["score_min"]) / (df["score_max"] - df["score_min"])

    return df  # Return the updated dataframe


# ### Extract Basic Length-Based Features from Text

# In[ ]:


def extract_length_features(text):
    """
    Extract basic length-related features from a given text, including counts and averages.

    Parameters:
        text (str): The raw essay text.

    Returns:
        dict: A dictionary with the number of sentences, number of words,
              average sentence length, and average word length.
    """
    # Split text into sentences using punctuation (., !, ?) as delimiters
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]  # Remove empty or whitespace-only sentences

    # Extract all word-like tokens (alphanumeric) from the text
    words = re.findall(r'\w+', text)

    # Count the number of sentences and words
    num_sentences = len(sentences)
    num_words = len(words)

    # Calculate average sentence length (in words), avoid division by zero
    avg_sentence_length = num_words / num_sentences if num_sentences > 0 else 0

    # Calculate average word length (in characters), avoid division by zero
    avg_word_length = sum(len(word) for word in words) / len(words) if words else 0

    return {
        "num_sentences": num_sentences,
        "num_words": num_words,
        "avg_sentence_length": avg_sentence_length,
        "avg_word_length": avg_word_length
    }


# ### Extract Readability Metrics from Text

# In[ ]:


def extract_readability_features(text):
    """
    Compute standard readability scores for a given text.

    These scores help estimate how difficult a text is to read, 
    often reflecting the education level required to understand it.

    Parameters:
        text (str): The input essay or paragraph.

    Returns:
        dict: A dictionary of various readability metrics.
    """
    return {
        "flesch_reading_ease": textstat.flesch_reading_ease(text),              
        # Higher is easier to read; typically ranges from 0 (hard) to 100 (easy)

        "flesch_kincaid_grade": textstat.flesch_kincaid_grade(text),            
        # U.S. school grade level required to understand the text

        "gunning_fog": textstat.gunning_fog(text),                              
        # Estimates years of formal education needed

        "smog_index": textstat.smog_index(text),                                
        # Focuses on polysyllabic words; suitable for short texts

        "dale_chall": textstat.dale_chall_readability_score(text),              
        # Considers difficult words based on a predefined list

        "automated_readability": textstat.automated_readability_index(text)     
        # Based on characters per word and words per sentence
    }


# ### Extract Text Complexity Features Using Dependency Parsing

# In[ ]:


def extract_text_complexity_features(text):
    """
    Analyze the syntactic structure of text to extract complexity-related features
    using dependency parse trees.

    Parameters:
        text (str): The input text to analyze.

    Returns:
        dict: A dictionary containing:
              - average parse tree depth
              - average leaf node depth
              - number of clauses
              - average clause length
              - maximum clause length
    """
    doc = nlp(text)  # Use spaCy to tokenize and parse the text

    tree_depths = []     # Stores depth of each sentence's parse tree
    leaf_depths = []     # Stores depth of each leaf node in the tree
    clause_counts = []   # Stores number of clauses per sentence
    clause_lengths = []  # Stores lengths of individual clauses

    for sent in doc.sents:
        # Find the root of the dependency tree for the sentence
        roots = [token for token in sent if token.head == token] 
        if not roots:
            continue
        root = roots[0]  

        # Recursive function to calculate depth of a token's subtree
        def get_depth(token):
            if not list(token.children):
                return 1
            return 1 + max(get_depth(child) for child in token.children)

        # Recursive function to collect depths of all leaf nodes
        def get_leaf_depths(token, depth=1):
            if not list(token.children):
                return [depth]
            return sum([get_leaf_depths(child, depth + 1) for child in token.children], [])

        tree_depths.append(get_depth(root))             # Depth of full sentence tree
        leaf_depths += get_leaf_depths(root)            # Collect all leaf depths

        # Count specific types of subordinate clauses
        clauses = [tok for tok in sent if tok.dep_ in ("ccomp", "advcl", "acl", "relcl", "xcomp", "mark")]
        clause_counts.append(len(clauses))              # Number of clauses in the sentence

        # Calculate number of tokens in each clause
        clause_lengths.extend([len(list(clause.subtree)) for clause in clauses])

    # Return summary statistics capturing syntactic complexity
    return {
        "avg_parse_tree_depth": sum(tree_depths) / len(tree_depths) if tree_depths else 0,
        "avg_leaf_depth": sum(leaf_depths) / len(leaf_depths) if leaf_depths else 0,
        "num_clauses": sum(clause_counts),
        "avg_clause_length": sum(clause_lengths) / len(clause_lengths) if clause_lengths else 0,
        "max_clause_length": max(clause_lengths) if clause_lengths else 0,
    }


# ### Extract Linguistic Variation Features from Text

# In[ ]:


def extract_text_variation_features(text):
    """
    Compute features that capture vocabulary richness and grammatical diversity.

    These metrics reflect how varied the writing is in terms of vocabulary and part-of-speech usage.

    Parameters:
        text (str): The input essay text.

    Returns:
        dict: A dictionary with:
              - type-token ratio
              - lexical density
              - number of unique POS tags
              - POS entropy
              - repetition rate
    """
    doc = nlp(text)  # Tokenize and tag using spaCy

    # Lowercase alphabetic words (filtering out punctuation/numbers)
    words = [token.text.lower() for token in doc if token.is_alpha]
    unique_words = set(words)

    # Collect part-of-speech (POS) tags
    pos_tags = [token.pos_ for token in doc if token.is_alpha]
    pos_counts = Counter(pos_tags)

    # Count how many words are content-bearing (e.g., nouns, verbs, adjectives, adverbs)
    content_tags = {"NOUN", "VERB", "ADJ", "ADV"}
    content_count = sum(1 for token in doc if token.pos_ in content_tags)

    # Compute entropy of POS tag distribution to measure grammatical diversity
    total_pos = sum(pos_counts.values())
    pos_entropy = -sum(
        (count / total_pos) * math.log2(count / total_pos)
        for count in pos_counts.values()
    ) if total_pos > 0 else 0

    return {
        "type_token_ratio": len(unique_words) / len(words) if words else 0,          
        # Measures vocabulary diversity (higher = more diverse)

        "lexical_density": content_count / len(words) if words else 0,              
        # Proportion of content words (vs. function words)

        "num_unique_pos_tags": len(pos_counts),                                      
        # Number of distinct POS types used

        "pos_entropy": pos_entropy,                                                  
        # Reflects how evenly POS types are used (higher = more varied grammar)

        "repeat_rate": (len(words) - len(unique_words)) / len(words) if words else 0 
        # Measures word repetition (lower = more unique word usage)
    }


# ### Extract Sentiment-Based Features from Text

# In[ ]:


def extract_sentiment_features(text):
    """
    Analyze the sentiment of the text using TextBlob.

    Parameters:
        text (str): The input text.

    Returns:
        dict: A dictionary containing:
              - sentiment polarity: how positive or negative the text is
              - sentiment subjectivity: how subjective or opinionated the text is
    """
    blob = TextBlob(text)  # Convert text into a TextBlob object for analysis

    polarity = blob.sentiment.polarity       
    # Polarity ranges from -1 (very negative) to 1 (very positive)

    subjectivity = blob.sentiment.subjectivity  
    # Subjectivity ranges from 0 (objective/factual) to 1 (highly subjective)

    return {
        "sent_polarity": polarity,
        "sent_subjectivity": subjectivity
    }


# ### Add Normalized Feature Set to DataFrame (Per Prompt)

# In[ ]:


def add_features(df, func, prefix):
    """
    Apply a feature-extraction function to each essay and normalize the resulting features
    per essay prompt (essay_set). Appends the normalized features back to the dataframe.

    Parameters:
        df (pd.DataFrame): Original dataframe containing at least 'essay' and 'essay_set' columns.
        func (function): A function that extracts features from a single essay (returns a dict).
        prefix (str): A string prefix to add to the new feature column names.

    Returns:
        pd.DataFrame: The input dataframe with new normalized feature columns added.
    """
    # Apply the feature extraction function to each essay
    type_feats = df["essay"].apply(func)

    # Convert the list of dictionaries into a DataFrame
    type_df = pd.DataFrame(list(type_feats))
    type_df["essay_set"] = df["essay_set"]  # Keep track of which prompt each essay belongs to

    scalers = {}       # Will store a separate scaler for each prompt
    normalized = []    # Will store the normalized feature sub-dataframes for each prompt

    for prompt in df["essay_set"].unique():
        # Select only the rows for the current prompt, excluding the prompt label
        sub = type_df[type_df["essay_set"] == prompt].drop(columns="essay_set")

        # Fit a MinMaxScaler on this subset and transform the features to range [0, 1]
        scaler = MinMaxScaler()
        norm_sub = scaler.fit_transform(sub)

        # Store the normalized DataFrame (retain original index)
        normalized.append(pd.DataFrame(norm_sub, columns=sub.columns, index=sub.index))

        # Save the scaler in case it's needed later (e.g., for inference)
        scalers[prompt] = scaler

    # Combine all normalized feature DataFrames and restore original ordering
    type_df_normalized = pd.concat(normalized).sort_index()

    # Add the normalized features to the original DataFrame with a prefix
    df = pd.concat([df, type_df_normalized.add_prefix(prefix)], axis=1)

    return df


# ### Convert Normalized Scores Back to Original Scale

# In[ ]:


def denormalize_score(normalized_score, df):
    """
    Convert normalized scores (ranging from 0 to 1) back to their original scale 
    using the min and max score values from the dataframe.

    Parameters:
        normalized_score (np.array or list): Normalized scores between 0 and 1.
        df (pd.DataFrame): DataFrame containing 'score_min' and 'score_max' columns.

    Returns:
        np.array: Denormalized scores rounded to the nearest integer.
    """
    normalized_score = np.squeeze(normalized_score)  # Ensure 1D array format if needed

    # Retrieve original min and max scores (per sample) from the dataframe
    min_score = df["score_min"].values
    max_score = df["score_max"].values

    # Scale normalized scores back to the original range
    denormalized_score = normalized_score * (max_score - min_score) + min_score

    return np.round(denormalized_score).astype(int)  # Round and convert to integers


# ### Generate 7-Train / 1-Test Prompt Splits (Cross-Prompt Setup)

# In[ ]:


def generate_train7_test1_splits(all_sets=None):
    """
    Generate 8-fold cross-prompt splits where each essay set (prompt) is used once as the test set,
    and the remaining 7 sets are used for training.

    Parameters:
        all_sets (list, optional): A list of all essay set IDs (defaults to sets 1–8).

    Returns:
        list of dict: A list containing 8 dictionaries, each with a 'train' list and a 'test' ID.
    """
    if all_sets is None:
        all_sets = [1, 2, 3, 4, 5, 6, 7, 8]  # Default to 8 essay prompts if not provided

    splits = []  # Will hold all (train, test) split configurations

    for test_set in all_sets:
        # Use all sets except the current one for training
        train_sets = [s for s in all_sets if s != test_set]

        # Add this split to the list
        splits.append({"train": train_sets, "test": test_set})

    return splits


# ### Preprocess Example for Tokenization and Feature Extraction

# In[ ]:


def preprocess(example, tokenizer):
    """
    Tokenize essay text and extract handcrafted features for model input.

    This function prepares each input example by:
      - Tokenizing the essay using the given tokenizer
      - Attaching the normalized score as the label
      - Collecting handcrafted features (e.g., length, readability, etc.)

    Parameters:
        example (dict): A single data sample with keys like 'essay', 'normalized_score', and feature columns.
        tokenizer (Tokenizer): A Hugging Face tokenizer (e.g., BERT tokenizer).

    Returns:
        dict: A dictionary containing:
              - tokenized inputs (input_ids, attention_mask, etc.)
              - 'labels': the normalized score
              - 'features': a list of handcrafted feature values
    """
    # Tokenize the essay with truncation and padding for fixed-length inputs
    tokens = tokenizer(example["essay"], truncation=True, padding="max_length", max_length=512)

    # Assign the normalized essay score as the label
    tokens["labels"] = example["normalized_score"]

    # Define prefixes for handcrafted feature columns
    feature_prefixes = ("len_", "read_", "comp_", "var_", "sent_")

    # Collect feature values that match the desired prefixes
    handcrafted_feats = [value for key, value in example.items() if key.startswith(feature_prefixes)]

    # Add handcrafted features to the tokenized output
    tokens["features"] = handcrafted_feats

    return tokens

