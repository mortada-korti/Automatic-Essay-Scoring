#!/usr/bin/env python
# coding: utf-8

# # BERT_utils_1.py

# ### Import Core BERT Components and Dependencies

# In[ ]:


# These bring in ready-made BERT components and utilities from Hugging Face's Transformers library.
from transformers.models.bert.modeling_bert import (
    BertSelfAttention,  # The multi-head self-attention mechanism used in BERT layers
    BertSelfOutput,     # Post-attention dense + dropout + residual connection
    BertEmbeddings,     # Word, position, and segment embeddings for BERT inputs
    BertPooler          # The pooling layer that produces a fixed-size vector from BERT's output
)

from transformers import BertModel, BertPreTrainedModel
# BertModel        → Full pretrained BERT architecture (encoder only)
# BertPreTrainedModel → Base class that handles loading, saving, and config management

import torch  # PyTorch library for building and training neural networks


# ### Mixture-of-Experts Feed-Forward Layer for BERT

# In[ ]:


# --- MoEFeedForward ---
# Purpose:
# Implements a "Mixture of Experts" feed-forward network.
# Each token's representation is processed by multiple expert networks.
# A small gating network decides how much each expert contributes (top-k selection possible).

class MoEFeedForward(torch.nn.Module):
    def __init__(self, hidden_dim, intermediate_dim, num_experts=7, dropout=0.2, top_k=2):
        super().__init__()
        self.hidden_dim = hidden_dim                # Input/output feature size
        self.intermediate_dim = intermediate_dim    # Hidden size inside each expert
        self.num_experts = num_experts              # Total experts available
        self.top_k = top_k                          # Number of experts to use per token

        # Gate network: predicts weights for each expert based on input
        self.gate = torch.nn.Linear(hidden_dim, num_experts)

        # Dropout to reduce overfitting
        self.dropout = torch.nn.Dropout(dropout)

        # Create the list of expert feed-forward networks
        self.experts = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(hidden_dim, intermediate_dim),  # First dense layer
                torch.nn.GELU(),                                # Activation function
                torch.nn.Linear(intermediate_dim, hidden_dim)   # Second dense layer
            ) for expert in range(num_experts)
        ])

    def forward(self, x, expert_mask=None):     
        # Get gate scores for each expert (before softmax)
        gate_logits = self.gate(x)

        if 0 < self.top_k < self.num_experts:
            # Select the top-k experts per token
            topk_values, topk_indices = torch.topk(gate_logits, self.top_k, dim=-1)

            # Initialize all as -inf (so non-top-k experts get zero weight after softmax)
            masked = torch.full_like(gate_logits, float("-inf"))

            # Fill only top-k positions with their scores
            masked.scatter_(-1, topk_indices, topk_values)

            # Stabilize values by subtracting the max in each row
            masked = masked - masked.amax(dim=-1, keepdim=True)

            # Softmax to turn scores into probabilities
            gate_weights = torch.nn.functional.softmax(masked, dim=-1)
        else:
            # Use all experts — just stabilize then softmax
            logits = gate_logits - gate_logits.amax(dim=-1, keepdim=True)
            gate_weights = torch.nn.functional.softmax(logits, dim=-1)

        # Run all experts and stack their outputs: shape (batch, seq_len, num_experts, hidden_dim)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=2)

        # Weighted sum of experts based on gate probabilities
        output = (expert_outputs * gate_weights.unsqueeze(-1)).sum(dim=2)

        # Apply dropout and return both the output and the gate weights
        return self.dropout(output), gate_weights


# ### Custom BERT Layer with Mixture-of-Experts (MoE) Feed-Forward

# In[ ]:


# --- BertLayerWithMoE ---
# Purpose:
# A modified BERT encoder layer where the usual feed-forward network (FFN)
# is replaced with the Mixture of Experts (MoE) feed-forward layer.
# Still keeps the standard BERT self-attention, residual connections, and layer normalization.

class BertLayerWithMoE(torch.nn.Module):
    def __init__(self, config, num_experts=7, top_k=2):
        super().__init__()

        # Standard multi-head self-attention module
        self.attention = BertSelfAttention(config)

        # Output stage after attention: dense + dropout + residual
        self.attention_output = BertSelfOutput(config)

        # Replace the standard FFN with the MoE FFN
        self.intermediate = MoEFeedForward(
            num_experts=num_experts,                     # total experts
            hidden_dim=config.hidden_size,               # input/output size
            dropout=config.hidden_dropout_prob,          # dropout rate
            intermediate_dim=config.intermediate_size,   # hidden size in each expert
            top_k=top_k                                  # how many experts to pick per token
        )

        # Dropout after MoE output
        self.output_dropout = torch.nn.Dropout(config.hidden_dropout_prob)

        # Layer normalization after combining MoE output with attention output
        self.output_norm = torch.nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def initialize_experts_from_ffn(self, pretrained_ffn):
        # Copy weights from an existing (pretrained) FFN into all experts
        for expert in self.intermediate.experts:
            # First linear layer weights & bias
            expert[0].weight.data.copy_(pretrained_ffn[0].weight.data.clone())
            expert[0].bias.data.copy_(pretrained_ffn[0].bias.data.clone())

            # Second linear layer weights & bias
            expert[2].weight.data.copy_(pretrained_ffn[2].weight.data.clone())
            expert[2].bias.data.copy_(pretrained_ffn[2].bias.data.clone())

            # Add small random noise so experts can diversify during training
            noise_scale = 0.0 if len(self.intermediate.experts) == 1 else 0.0
            for param in expert.parameters():
                param.data.add_(noise_scale * torch.randn_like(param))

    def forward(self, hidden_states, attention_mask=None, expert_mask=None):
        # Apply self-attention to input
        attention_output = self.attention(hidden_states, attention_mask)[0]

        # Apply attention output projection + dropout + residual
        attention_output = self.attention_output(attention_output, hidden_states)

        # Pass through the MoE feed-forward network
        moe_output, gate_weights = self.intermediate(attention_output, expert_mask=expert_mask)

        # Apply dropout to MoE output
        ffn_output = self.output_dropout(moe_output)

        # Add residual connection from attention output and normalize
        layer_output = self.output_norm(ffn_output + attention_output)

        # Return the final layer output and the expert gate weights
        return layer_output, gate_weights


# ### MoeBERTModel: Full BERT Encoder with MoE-Enhanced Layers

# In[ ]:


# --- MoeBERTModel ---
# Purpose:
# A custom BERT model where every transformer layer's feed-forward network (FFN)
# is replaced by a Mixture-of-Experts (MoE) version.
# Starts from a pretrained BERT, reuses its embeddings, pooler, and attention layers,
# then initializes MoE experts from the pretrained FFN weights.

class MoeBERTModel(BertPreTrainedModel):
    def __init__(self, config, num_experts=7, top_k=2, pretrained_name_or_path="bert-base-uncased"):
        super().__init__(config)
        self.config = config
        self.num_experts = num_experts  # total experts per layer
        self.top_k = top_k              # number of active experts per token

        # Load pretrained BERT model (for initialization)
        base_bert = BertModel.from_pretrained(pretrained_name_or_path, config=config)
        base_bert.eval()  # disable dropout during weight copying

        # Reuse pretrained embedding and pooler layers
        self.embeddings = base_bert.embeddings
        self.pooler = base_bert.pooler

        # Create MoE-based transformer layers
        self.encoder_layers = torch.nn.ModuleList([
            BertLayerWithMoE(config, num_experts=num_experts, top_k=top_k)
            for _ in range(config.num_hidden_layers)
        ])

        # Initialize MoE layers from the pretrained FFN weights
        with torch.no_grad():
            for i, moe_layer in enumerate(self.encoder_layers):
                base_layer = base_bert.encoder.layer[i]

                # Reuse attention sublayers directly from pretrained BERT
                moe_layer.attention = base_layer.attention.self
                moe_layer.attention_output = base_layer.attention.output

                # Copy LayerNorm weights from pretrained
                moe_layer.output_norm.weight.copy_(base_layer.output.LayerNorm.weight)
                moe_layer.output_norm.bias.copy_(base_layer.output.LayerNorm.bias)

                # Create a reference FFN sequence from pretrained layer
                pretrained_ffn = torch.nn.Sequential(
                    base_layer.intermediate.dense,
                    torch.nn.GELU(),
                    base_layer.output.dense
                )

                # Initialize each MoE expert from the pretrained FFN
                moe_layer.initialize_experts_from_ffn(pretrained_ffn)

        # Store last gate weights for inspection
        self._last_gate_weights = None

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, expert_mask=None):
        device = input_ids.device
        extended_attention_mask = None

        # Prepare the attention mask for BERT (broadcasts to all heads)
        if attention_mask is not None:
            extended_attention_mask = self.get_extended_attention_mask(
                attention_mask, input_ids.shape, device
            )

        # Move expert mask to correct device if provided
        if expert_mask is not None:
            expert_mask = expert_mask.to(device)

        # Run the embedding layer
        hidden_states = self.embeddings(input_ids=input_ids, token_type_ids=token_type_ids)

        gate_weights_list = []  # store gating info from each layer

        # Pass through all MoE transformer layers
        for layer in self.encoder_layers:
            hidden_states, gate_weights = layer(
                hidden_states,
                attention_mask=extended_attention_mask,
                expert_mask=expert_mask
            )
            gate_weights_list.append(gate_weights)

        # Pooling: mean over valid tokens if mask provided, else use [CLS] token
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)  # (B, T, 1)
            summed = (hidden_states * mask).sum(dim=1)                   # sum over tokens
            denom = mask.sum(dim=1).clamp_min(1e-6)                      # avoid div by zero
            pooled_output = summed / denom
        else:
            pooled_output = hidden_states[:, 0]  # CLS token

        # Save gating info for later inspection
        self._last_gate_weights = gate_weights_list

        return hidden_states, pooled_output, gate_weights_list


# ### MoeBERTScorer: Essay Scoring Head with Optional Handcrafted Features & MoE Regularization

# In[ ]:


# --- MoeBERTScorer ---
# Purpose:
# Wraps a MoeBERTModel to produce a single scalar score (e.g., for regression tasks).
# Can take optional extra features, track expert usage stats, and apply extra loss terms
# to supervise or regularize expert gating behavior.

class MoeBERTScorer(torch.nn.Module):
    def __init__(self, base_model: MoeBERTModel, dropout=0.2, feature_dim=0):
        super().__init__()
        self.encoder = base_model                # The underlying BERT+MoE model
        self.feature_dim = feature_dim            # Size of extra input features

        # Total input size to regressor = BERT output size + any extra feature size
        input_dim = self.encoder.config.hidden_size + feature_dim

        # Simple regressor: linear projection to 1 value + dropout
        self.regressor = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 1),
            torch.nn.Dropout(dropout)
        )

        # For logging/debugging expert behavior
        self.last_gate_weights = None
        self.expert_usage_counts = None
        self.expert_entropy = None

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        features=None,
        labels=None,
        aux_loss_weight=0.5,
        expert_mask=None
    ):
        # Encode inputs using the MoE BERT model
        hidden_states, pooled_output, gate_weights_list = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            expert_mask=expert_mask
        )

        # Last layer's gating probabilities
        last_gate = gate_weights_list[-1] 

        # ---- Optional: Track expert usage + entropy ----
        if aux_loss_weight is not None and aux_loss_weight > 0 and last_gate.ndim == 3:
            with torch.no_grad():
                prob_mass = last_gate.sum(dim=(0, 1))  # total usage per expert across batch/tokens
                self.expert_usage_counts = prob_mass.detach().cpu()

                pm_sum = prob_mass.sum()
                if pm_sum > 0:
                    # Normalize to distribution and compute entropy
                    prob_dist = (prob_mass / (pm_sum + 1e-12)).clamp_min(1e-12)
                    self.expert_entropy = float(-(prob_dist * prob_dist.log()).sum().item())
                else:
                    self.expert_entropy = 0.0

        # Save average gate weights for inspection
        with torch.no_grad():
            self.last_gate_weights = last_gate.mean(dim=1).detach().cpu()

        # ---- Append extra features if provided ----
        if features is not None:
            if features.device != pooled_output.device:
                features = features.to(pooled_output.device)
            if features.dtype != pooled_output.dtype:
                features = features.to(pooled_output.dtype)
            pooled_output = torch.cat([pooled_output, features], dim=-1)

        # ---- Predict score ----
        score = self.regressor(pooled_output).squeeze(-1)  # (B,)

        # ---- Loss computation ----
        loss = None
        aux_loss = None

        if labels is not None:
            labels = labels.to(score.dtype)

            # Base regression loss
            loss = torch.nn.functional.mse_loss(score, labels)

            # Auxiliary entropy regularization loss (discourage low-entropy gate usage)
            if gate_weights_list and aux_loss_weight is not None and aux_loss_weight > 0:
                gate_weights = gate_weights_list[-1]  
                mean_gates = gate_weights.mean(dim=(0, 1))  
                mean_gates = mean_gates / (mean_gates.sum() + 1e-12)
                mean_gates = mean_gates.clamp_min(1e-12)
                entropy = -(mean_gates * mean_gates.log()).sum()
                aux_loss = -entropy  
                loss = loss + aux_loss_weight * aux_loss

        # ---- Return outputs ----
        return {
            "loss": loss,                    # total loss (if labels given)
            "logits": score,                  # predicted scores
            "hidden_states": hidden_states,   # encoder hidden states
            "aux_loss": aux_loss if labels is not None else None
        }


# In[ ]:


# --- preprocess ---
# Purpose:
# Converts a raw essay example into tokenized BERT inputs plus extra features and labels.
# - Tokenizes the essay text.
# - Adds the normalized score as the regression label.
# - Collects handcrafted numerical features.
# - Keeps track of which essay set it belongs to.

def preprocess(example, tokenizer):
    # Tokenize the essay text with fixed length (truncate/pad to 512 tokens)
    tokens = tokenizer(
        example["essay"],
        truncation=True,
        padding="max_length",
        max_length=512,
    )

    # Add the normalized score as the label for training
    tokens["labels"] = float(example["normalized_score"])

    # Prefixes for handcrafted features we want to extract
    feature_prefixes = ("len_", "read_", "comp_", "var_", "sent_")

    # Collect all (key, value) pairs that start with those prefixes
    feat_items = [(k, v) for k, v in example.items() if k.startswith(feature_prefixes)]

    # Sort features by their key name to ensure consistent ordering
    feat_items.sort(key=lambda kv: kv[0])  

    # Convert feature values to floats
    handcrafted_feats = [float(v) for _, v in feat_items]

    # Store features in tokens dictionary
    tokens["features"] = handcrafted_feats

    # Store essay set ID as an integer
    tokens["essay_set"] = int(example["essay_set"])

    return tokens


# In[ ]:


# --- map_essay_set_to_expert ---
# Purpose:
# Creates a mapping from essay set IDs to expert indices.
# Ensures mapping is consistent by sorting the set IDs first.

def map_essay_set_to_expert(train_sets):
    # Enumerate over sorted essay set IDs and assign each one a unique index
    return {eid: idx for idx, eid in enumerate(sorted(train_sets))}


# In[ ]:


# --- add_expert_mask ---
# Purpose:
# Assigns an expert index to an example based on its essay set ID.
# If the essay set is not in the map, assigns None.

def add_expert_mask(example, expert_map):
    # Get essay set ID (or None if missing)
    es = example.get("essay_set", None)

    # If the essay set exists in the map, use its expert index
    if es in expert_map:
        example["expert_mask"] = expert_map[es]
    else:
        # If not found, mark expert mask as None
        example["expert_mask"] = None

    return example


# In[ ]:


# --- data_collator ---
# Purpose:
# Custom batch collation function for Hugging Face Trainer.
# Converts a list of feature dictionaries into a batch of PyTorch tensors,
# handling both standard BERT inputs and additional custom fields.

def data_collator(features):
    batch = {}  # will store the batched tensors

    # Batch token IDs if present
    if "input_ids" in features[0]:
        batch["input_ids"] = torch.as_tensor([f["input_ids"] for f in features], dtype=torch.long)

    # Batch attention masks if present
    if "attention_mask" in features[0]:
        batch["attention_mask"] = torch.as_tensor([f["attention_mask"] for f in features], dtype=torch.long)

    # Batch token type IDs if present and not None
    if "token_type_ids" in features[0] and features[0]["token_type_ids"] is not None:
        batch["token_type_ids"] = torch.as_tensor([f["token_type_ids"] for f in features], dtype=torch.long)

    # Batch labels (float for regression)
    if "labels" in features[0]:
        batch["labels"] = torch.as_tensor([f["labels"] for f in features], dtype=torch.float32)

    # Batch handcrafted features if they exist and are non-empty
    if "features" in features[0] and features[0]["features"] is not None:
        first_feats = features[0]["features"]
        if isinstance(first_feats, (list, tuple)) and len(first_feats) > 0:
            batch["features"] = torch.as_tensor([f["features"] for f in features], dtype=torch.float32)

    # Batch expert mask if all examples have it
    if "expert_mask" in features[0]:
        mask_vals = [f["expert_mask"] for f in features]
        if all(m is not None for m in mask_vals):
            batch["expert_mask"] = torch.as_tensor(mask_vals, dtype=torch.long)

    return batch


# ### Freeze BERT Layers Except the Last Few (Fine-Tuning Strategy)

# In[ ]:


# --- freeze_bert_layers ---
# Purpose:
# Freezes all model parameters except for:
# - The last `num_unfrozen` encoder layers
# - The pooler (if present)
# - The regressor head (if present)
# Useful for fine-tuning only the top layers while keeping most of BERT fixed.

def freeze_bert_layers(model, num_unfrozen=2):
    # Freeze all parameters in the model
    for p in model.parameters():
        p.requires_grad = False

    # Get the encoder module (may be directly the model or inside .encoder)
    encoder = getattr(model, "encoder", model)

    # Unfreeze the last `num_unfrozen` encoder layers
    if hasattr(encoder, "encoder_layers"):
        total_layers = len(encoder.encoder_layers)
        start = max(0, total_layers - int(num_unfrozen))
        for i in range(start, total_layers):
            for p in encoder.encoder_layers[i].parameters():
                p.requires_grad = True

    # Unfreeze the pooler if it exists
    if hasattr(encoder, "pooler") and encoder.pooler is not None:
        for p in encoder.pooler.parameters():
            p.requires_grad = True

    # Unfreeze the regressor head if it exists
    if hasattr(model, "regressor") and model.regressor is not None:
        for p in model.regressor.parameters():
            p.requires_grad = True

