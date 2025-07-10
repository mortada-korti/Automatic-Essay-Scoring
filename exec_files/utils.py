#!/usr/bin/env python
# coding: utf-8

# # Utils.py script

# ## Import Required Libraries

# In[ ]:


from transformers import AutoTokenizer, AutoModel, get_scheduler, Trainer, TrainingArguments 
from transformers.modeling_outputs import SequenceClassifierOutput
from sklearn.metrics import mean_squared_error, cohen_kappa_score  
from torch.utils.data import TensorDataset 
from tqdm import tqdm  
import pandas as pd 
import numpy as np  
import random  
import torch  


# In[ ]:


def generate_cross_prompt_splits(all_sets=None, val_set=2):
    '''
    Generates multiple cross-prompt data splits.
    Each split uses:
      - a fixed validation set (val_set),
      - one prompt as the test set,
      - the remaining prompts as the training set.
    Returns a list of dictionaries with 'train', 'val', and 'test' keys.
    '''
    if all_sets is None:
        all_sets = [1, 2, 3, 4, 5, 6, 7, 8]

    splits = []
    for test_set in all_sets:
        if test_set == val_set:
            continue  

        train_sets = [s for s in all_sets if s not in [val_set, test_set]]
        splits.append({"train": train_sets, "val": val_set, "test": test_set})
    return splits


# In[ ]:


def get_split(df, train_sets, val_set, test_set):
    '''
    Splits the full dataset into train, validation, and test DataFrames
    based on the provided essay set IDs.
    
    Returns:
        train_df: DataFrame containing essays from train_sets
        val_df:   DataFrame containing essays from val_set
        test_df:  DataFrame containing essays from test_set
    '''
    train_df = df[df["essay_set"].isin(train_sets)].copy()
    val_df = df[df["essay_set"] == val_set].copy()
    test_df = df[df["essay_set"] == test_set].copy()
    return train_df, val_df, test_df


# In[ ]:


class RegressionTrainer(Trainer):
    '''
    Custom Trainer subclass for regression tasks (like essay scoring).
    Overrides the default classification loss with Mean Squared Error (MSE).
    '''
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        '''
        Computes regression loss using MSE between predicted and true scores.

        Args:
            model: the model being trained
            inputs: input dictionary including 'labels'
            return_outputs: if True, returns both loss and model outputs

        Returns:
            loss (and optionally outputs)
        '''
        print("inputs = ", inputs)
        
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        preds = outputs.squeeze()
        loss = torch.nn.functional.mse_loss(preds, labels.float())
        return (loss, outputs) if return_outputs else loss


# In[ ]:


def compute_metrics_factory(df):
    '''
    Creates a custom evaluation function for use with Hugging Face Trainer.
    This function:
      - denormalizes predictions and labels back to original score scale
      - computes Quadratic Weighted Kappa (QWK) between them

    Returns:
        compute_metrics: a function that takes (predictions, labels) and returns {"qwk": value}
    '''
    def compute_metrics(eval_preds):
        predictions, labels = eval_preds
        predictions = np.squeeze(np.asarray(predictions))
        labels = np.squeeze(np.asarray(labels))
        preds = denormalize_score(predictions, df)
        labels = denormalize_score(labels, df)
        qwk = cohen_kappa_score(preds, labels, weights="quadratic")
        return {"qwk": qwk}
    return compute_metrics


# In[ ]:


def denormalize_score(normalized_score, df):
    '''
    Converts normalized scores (in range [0, 1]) back to their original score scale
    using the min and max score values for each essay.

    Args:
        normalized_score: array-like of normalized predictions or labels
        df: DataFrame containing 'score_min' and 'score_max' columns

    Returns:
        denormalized_score: array of rounded integer scores in original scale
    '''
    normalized_score = np.squeeze(normalized_score)
    min_score = df["score_min"].values
    max_score = df["score_max"].values
    denormalized_score = normalized_score * (max_score - min_score) + min_score
    return np.round(denormalized_score).astype(int)


# In[ ]:


def freeze_bert_layers(model, num_unfrozen=2):
    '''
    Freezes all layers of a BERT-based model except for the last `num_unfrozen` encoder layers
    and the pooler (if it exists). This is useful for partial fine-tuning.

    Args:
        model: the BERT model (e.g., model.bert) whose layers are to be frozen
        num_unfrozen: number of top encoder layers to keep trainable (default is 2)

    Behavior:
        - Freezes all model parameters by default
        - Unfreezes only the last N transformer layers
        - Also unfreezes the pooler if present
    '''
    for param in model.parameters():
        param.requires_grad = False

    if hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
        total_layers = len(model.encoder.layer)
        for i in range(total_layers - num_unfrozen, total_layers):
            for param in model.encoder.layer[i].parameters():
                param.requires_grad = True

    if hasattr(model, "pooler"):  
        for param in model.pooler.parameters():
            param.requires_grad = True


# In[ ]:


class EssayScoringHead(torch.nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.regressor = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, 1),    
            torch.nn.Dropout(0.2)
        )

    def forward(self, cls_vector):
        return self.regressor(cls_vector)


# In[ ]:


class EssayScoringModel(torch.nn.Module):
    def __init__(self, base_model_name):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model_name)
        self.head = EssayScoringHead(self.encoder.config.hidden_size)

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_vector = outputs.last_hidden_state[:, 0]
        score = self.head(cls_vector).squeeze(-1)  # shape: [batch_size]

        # compute loss inside the model
        loss = None
        if labels is not None:
            loss = torch.nn.functional.mse_loss(score, labels.float())

        # return as Hugging Face expects
        return SequenceClassifierOutput(
            loss=loss,
            logits=score.unsqueeze(-1),  # keep shape [batch_size, 1]
            hidden_states=outputs.hidden_states if hasattr(outputs, "hidden_states") else None,
            attentions=outputs.attentions if hasattr(outputs, "attentions") else None
        )


# In[ ]:


class MoERegressor(torch.nn.Module):
    def __init__(self, hidden_dim: int, num_experts: int, dropout_rate: float = 0.2):
        """
        Mixture-of-Experts Regressor with dropout.
        """
        super(MoERegressor, self).__init__()

        self.num_experts = num_experts
        self.dropout = torch.nn.Dropout(p=dropout_rate)  # Apply before experts and gate

        # K separate experts (linear regressors)
        self.experts = torch.nn.ModuleList([
            torch.nn.Linear(hidden_dim, 1) for _ in range(num_experts)
        ])

        # Gate network: outputs K trust logits
        self.gate = torch.nn.Linear(hidden_dim, num_experts)

    def forward(self, x):
        """
        Forward pass.
        x: (batch_size, hidden_dim) from BERT [CLS] vector
        """
        # Apply dropout to BERT representation
        x_dropped = self.dropout(x)

        # Pass through each expert
        expert_outputs = [expert(x_dropped) for expert in self.experts]  # List of [B, 1]
        expert_outputs = torch.stack(expert_outputs, dim=1).squeeze(-1)  # Shape: [B, K]

        # Gate prediction with dropout too
        gate_logits = self.gate(x_dropped)  # [B, K]
        gate_weights = torch.nn.functional.softmax(gate_logits, dim=1)  # [B, K]

        # Weighted sum of expert scores
        final_output = torch.sum(expert_outputs * gate_weights, dim=1)  # [B]

        return final_output, expert_outputs, gate_weights


# In[ ]:


def compute_auxiliary_loss(gate_weights, top_k=1):
    """
    Computes the auxiliary diversity loss for the gate weights (π).

    Parameters:
    - gate_weights: tensor of shape (batch_size, num_experts)
    - top_k: number of top overlaps to ignore when computing the penalty (default = 1)

    Returns:
    - aux_loss: scalar tensor (the diversity penalty)
    """
    batch_size, num_experts = gate_weights.shape

    # Step 1: Compute similarity matrix ΠᵀΠ → shape: (num_experts, num_experts)
    # This shows how similar each pair of experts is across the batch
    similarity_matrix = gate_weights.T @ gate_weights  # shape (K, K)

    # Step 2: Create identity matrix I of shape (K, K)
    identity_matrix = torch.eye(num_experts, device=gate_weights.device)

    # Step 3: Compute Frobenius norm of (ΠᵀΠ − I)
    # This penalizes overlap between experts' trust vectors
    numerator = torch.norm(similarity_matrix - identity_matrix, p='fro')

    # Step 4: Normalize it so the loss is between 0 and 1
    # Compute norm of (J − I), where J is all-ones matrix
    all_ones = torch.ones_like(similarity_matrix)
    denominator = torch.norm(all_ones - identity_matrix, p='fro')

    aux_loss = numerator / denominator

    return aux_loss


# In[ ]:


class EssayScoringMoEModel(torch.nn.Module):
    def __init__(self, base_model_name, num_experts=6, dropout=0.2):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model_name)

        # Pass dropout rate to MoERegressor
        self.head = MoERegressor(
            hidden_dim=self.encoder.config.hidden_size,
            num_experts=num_experts,
            dropout_rate=dropout 
        )

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        # Get BERT embeddings
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_vector = outputs.last_hidden_state[:, 0]  # [CLS] token

        # Run through MoE head
        final_score, expert_outputs, gate_weights = self.head(cls_vector)

        # Compute MSE loss (main)
        loss = None
        if labels is not None:
            mse_loss = torch.nn.functional.mse_loss(final_score, labels.float())
            aux_loss = compute_auxiliary_loss(gate_weights)
            alpha = getattr(self.encoder.config, "aux_loss_weight", 0.5)
            loss = mse_loss + alpha * aux_loss

        self.last_gate_weights = gate_weights.detach().cpu()  # save for access after inference
        
        return SequenceClassifierOutput(
            loss=loss,
            logits=final_score.unsqueeze(-1),  # [B, 1]
            hidden_states=outputs.hidden_states if hasattr(outputs, "hidden_states") else None,
            attentions=outputs.attentions if hasattr(outputs, "attentions") else None
        )


# In[ ]:


def generate_train7_test1_splits(all_sets=None):
    """
    Generate 8 splits: train on 7 essay sets, test on the remaining one.
    No validation set is used.
    """
    if all_sets is None:
        all_sets = [1, 2, 3, 4, 5, 6, 7, 8]

    splits = []
    for test_set in all_sets:
        train_sets = [s for s in all_sets if s != test_set]
        splits.append({"train": train_sets, "test": test_set})
    return splits


# ## Normalize scores within each essay set

# In[ ]:


def normalize_score(df):
    df.copy()
    df["score_min"] = df.groupby("essay_set")["domain1_score"].transform("min")
    df["score_max"] = df.groupby("essay_set")["domain1_score"].transform("max")
    df["normalized_score"] =  (df["domain1_score"] - df["score_min"]) / (df["score_max"] - df["score_min"])
    return df


# ## Evaluate the Model on Any Dataset

# In[ ]:


def eval_model(df_loader, df, device, model, regressor):
    model.eval()
    regressor.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in df_loader:
            preds, labels = get_preds(batch, device, model, regressor)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    mse = mean_squared_error(all_labels, all_preds)
    print(f"\nMean Squared Error: {mse:.4f}")

    score_min = df["score_min"].values
    score_max = df["score_max"].values
    essay_set = df["essay_set"].values
    raw_labels = df["domain1_score"].values
    
    denorm_preds = np.round(np.array(all_preds) * (score_max - score_min) + score_min).astype(int).tolist()
    original_labels = raw_labels.astype(int).tolist()
    denorm_preds = [int(p) for p in denorm_preds]

    df["prediction"] = denorm_preds
    df["true_score"] = original_labels

    overall_qwk, per_set_qwk = compute_qwk(df)
    print(f"QWK: {overall_qwk:.4f}")
    print("\nPer Essay Set QWK:")
    for set_id, qwk in per_set_qwk.items():
        print(f"Essay Set {set_id}: QWK = {qwk:.4f}")
    
    return overall_qwk


# ## Train model and return best weights

# In[ ]:


def train_model(train_loader, val_loader, val_df, device, model, regressor, num_epochs, learning_rate, warmup_ratio=0.1):
    model.train()
    regressor.train()

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(regressor.parameters()),
        lr=learning_rate
    )

    total_steps = len(train_loader) * num_epochs
    warmup_steps = int(warmup_ratio * total_steps)

    scheduler = get_scheduler(
        name="linear",  
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

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
            scheduler.step()
            optimizer.zero_grad()
            progress_bar.set_postfix({"Loss": loss.item()})
             
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
            
    return best_qwk, best_model_state, best_regressor_state, best_epoch, warmup_steps


# ## Get predictions from model

# In[ ]:


def get_preds(batch, device, model, regressor):
    input_ids, attention_mask, labels = [x.to(device) for x in batch]
    
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    cls_token_vector = outputs.last_hidden_state[:, 0, :]

    preds = torch.sigmoid(regressor(cls_token_vector)).squeeze()
    return preds, labels


# ## Encode text into input tensors

# In[ ]:


def encode_dataset(df, tokenizer, label_column="normalized_score"):
    encodings = tokenizer(df["essay"].tolist(), padding="max_length", truncation=True, max_length=512, return_tensors="pt")
    
    labels = torch.tensor(df[label_column].values, dtype=torch.float)
    dataset = TensorDataset(encodings.input_ids, encodings.attention_mask, labels)
    return dataset


# ## Set seeds for reproducibility

# In[ ]:


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ## Model Selection

# In[ ]:


def load_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return tokenizer, model


# In[ ]:


def load_results(csv_path):
    prev_df = pd.read_csv(csv_path)
    tried_configs = set(tuple(row) for row in prev_df[["learning_rate", "batch_size", "dropout"]].values)

    best_qwk = prev_df["val_qwk"].max()
    best_row = prev_df.loc[prev_df["qwk"].idxmax()]
    best_config = {"lr": best_row["learning_rate"], "bs": best_row["batch_size"], "dropout": best_row["dropout"]}
    best_epoch = int(best_row["epoch"])
    return best_qwk, best_row, best_config, best_epoch, tried_configs


# In[ ]:


def compute_qwk_(df):
    per_set_qwk = {}
    total_essays = 0

    for set_id in sorted(df["essay_set"].unique()):
        mask = df["essay_set"] == set_id
        y_true = df.loc[mask, "true_score"]
        y_pred = df.loc[mask, "prediction"]

        qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")
        count = len(y_true)
        per_set_qwk[set_id] = qwk
        
    overall_qwk = sum(per_set_qwk.values()) / len(per_set_qwk)
    return overall_qwk, per_set_qwk

