#!/usr/bin/env python
# coding: utf-8

# # Utils.py script

# ## Import Required Libraries

# In[1]:


from sklearn.metrics import mean_squared_error, cohen_kappa_score  # For evaluation metrics
from torch.utils.data import TensorDataset  # Helps create datasets for PyTorch
from tqdm import tqdm  # Visual progress bars for loops
import numpy as np  # For fast numerical operations (e.g., arrays, math)
import random  # For controlling randomness (e.g., shuffling, reproducibility)
import torch  # Core deep learning library


# ## Normalize scores within each essay set

# In[2]:


def normalize_score(df):
    df.copy()  # make a copy to avoid modifying original

    # Get min and max score for each essay_set (prompt)
    df["score_min"] = df.groupby("essay_set")["domain1_score"].transform("min")
    df["score_max"] = df.groupby("essay_set")["domain1_score"].transform("max")

    # Normalize score to range [0, 1] for each prompt
    df["normalized_score"] =  (df["domain1_score"] - df["score_min"]) / (df["score_max"] - df["score_min"])
    return df


# ## Evaluate the Model on Any Dataset

# In[3]:


def eval_model(df_loader, df, device, model, regressor):
    # Set models to evaluation mode (disable dropout, etc.)
    model.eval()
    regressor.eval()

    # variables to store original scores aand model's predictions
    all_preds = []
    all_labels = []

    # No gradient tracking needed for evaluation
    with torch.no_grad():
        for batch in df_loader:
            preds, labels = get_preds(batch, device, model, regressor)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Compute Mean Squared Error
    mse = mean_squared_error(all_labels, all_preds)
    print(f"\nMean Squared Error: {mse:.4f}")

    # Denormalize predictions to original score scale
    score_min = df["score_min"].values
    score_max = df["score_max"].values
    essay_set = df["essay_set"].values
    raw_labels = df["domain1_score"].values
    
    denorm_preds = np.round(np.array(all_preds) * (score_max - score_min) + score_min).astype(int).tolist()
    original_labels = raw_labels.astype(int).tolist()
    denorm_preds = [int(p) for p in denorm_preds]

    # Store predictions and true scores in the DataFrame
    df["prediction"] = denorm_preds
    df["true_score"] = original_labels

    # Compute QWK (overall and per essay set)
    qwk, per_set_qwk = compute_qwk(df)
    print(f"QWK: {qwk:.4f}")
    print("\nPer Essay Set QWK:")
    for set_id, qwk in per_set_qwk.items():
        print(f"Essay Set {set_id}: QWK = {qwk:.4f}")
    
    return qwk


# ## Compute Quadratic Weighted Kappa Metric

# In[4]:


def compute_qwk(df):
    per_set_qwk = {}
    total_essays = 0

    for set_id in sorted(df["essay_set"].unique()):
        mask = df["essay_set"] == set_id
        y_true = df.loc[mask, "true_score"]
        y_pred = df.loc[mask, "prediction"]

        qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")
        count = len(y_true)
        per_set_qwk[set_id] = qwk
        
    # Average QWK across all sets
    qwk = sum(per_set_qwk.values()) / len(per_set_qwk)
    return qwk, per_set_qwk


# ## Train model and return best weights

# In[5]:


def train_model(train_loader, val_loader, val_df, device, model, regressor, num_epochs, learning_rate=2e-5):
    model.train()
    regressor.train()

    # Loss and optimizer
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(regressor.parameters()),
        lr=learning_rate
    )

    # Early stopping setup
    best_qwk = -1
    patience = 3
    patience_counter = 0
    best_model_state = None
    best_regressor_state = None
    best_epoch = 0
    
    for epoch in range(1, num_epochs+1):
        progress_bar = tqdm(train_loader, desc=f"Training (Epoch {epoch})", leave=False)
        for step, batch in enumerate(progress_bar):
            preds, labels = get_preds(batch, device, model, regressor)
            loss = loss_fn(preds, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            progress_bar.set_postfix({"Loss": loss.item()})
        
        # Evaluate on validation set      
        qwk = eval_model(val_loader, val_df, device, model, regressor)

        if qwk > best_qwk:
            best_qwk = qwk
            patience_counter = 0
            best_model_state = model.state_dict()
            best_regressor_state = regressor.state_dict()
            best_epoch = epoch
            print(f"\nEpoch {epoch}: Best QWK for this config = {qwk:.4f}")
            
        else:
            patience_counter += 1
            print(f"\nEpoch {epoch}: QWK = {qwk:.4f} (no improvement)")
    
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered after {epoch} epochs.")
            break
            
    return best_qwk, best_model_state, best_regressor_state, best_epoch


# ## Get predictions from model

# In[6]:


def get_preds(batch, device, model, regressor):
    # Unpack and send to device
    input_ids, attention_mask, labels = [batch[0].to(device), batch[1].to(device), batch[2].to(device)]
    
    # Forward pass through BERT
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    cls_token_vector = outputs.last_hidden_state[:, 0, :]

    # Predict using regressor
    preds = torch.sigmoid(regressor(cls_token_vector)).squeeze()
    return preds, labels


# ## Encode text into input tensors

# In[7]:


def encode_dataset(df, tokenizer, label_column="normalized_score"):
    # Tokenize essays
    encodings = tokenizer(df["essay"].tolist(), padding="max_length", truncation=True, max_length=512, return_tensors="pt")
    
    # Create dataset with input ids, attention mask, and labels
    labels = torch.tensor(df[label_column].values, dtype=torch.float)
    dataset = TensorDataset(encodings.input_ids, encodings.attention_mask, labels)
    return dataset


# ## Set seeds for reproducibility

# In[8]:


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

