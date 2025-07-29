#!/usr/bin/env python
# coding: utf-8

# # DistilBERT_utils.py

# ### Import Core DistilBERT Components and Dependencies

# In[ ]:


from transformers.models.distilbert.modeling_distilbert import (
    Embeddings,            # Token + position embeddings for DistilBERT
    TransformerBlock       # Basic encoder block used in place of full-layer modules
)

from transformers import (
    DistilBertModel,        # Pretrained DistilBERT backbone
    DistilBertPreTrainedModel,  # Base class for custom heads
    DistilBertConfig        # Configuration for model structure
)

import torch            
import numpy as np         


# ### MoEFeedForward: Mixture-of-Experts Block for DistilBERT

# In[ ]:


class MoEFeedForward(torch.nn.Module):
    """
    A Mixture-of-Experts (MoE) feed-forward module used to replace or enhance
    the standard feed-forward layer in transformer blocks.

    Each input token is routed to a subset of expert networks, either densely or
    using top-k sparse gating.

    Parameters:
        hidden_dim (int): Input and output dimensionality.
        intermediate_dim (int): Hidden size inside each expert’s MLP.
        num_experts (int): Total number of expert networks.
        dropout (float): Dropout probability after combining expert outputs.
        top_k (int): Number of experts to activate per token.
    """
    def __init__(self, hidden_dim, intermediate_dim, num_experts=7, dropout=0.2, top_k=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.num_experts = num_experts
        self.top_k = top_k

        # Define each expert as a two-layer MLP with GELU activation
        self.experts = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(hidden_dim, intermediate_dim),
                torch.nn.GELU(),
                torch.nn.Linear(intermediate_dim, hidden_dim)
            ) for _ in range(num_experts)
        ])

        # Gating network to compute expert weights per token
        self.gate = torch.nn.Linear(hidden_dim, num_experts)

        # Dropout applied after expert aggregation
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        # Compute gating logits for all experts
        gate_logits = self.gate(x)  # (B, T, E)

        # Apply sparse top-k gating if configured
        if self.top_k > 0 and self.top_k < self.num_experts:
            topk_values, topk_indices = torch.topk(gate_logits, self.top_k, dim=-1)
            mask = torch.full_like(gate_logits, float('-inf'))
            mask.scatter_(-1, topk_indices, topk_values)
            gate_weights = torch.nn.functional.softmax(mask, dim=-1)
        else:
            gate_weights = torch.nn.functional.softmax(gate_logits, dim=-1)  # Dense routing

        # Compute each expert’s output
        expert_outputs = [expert(x) for expert in self.experts]     # List of (B, T, H)
        expert_outputs = torch.stack(expert_outputs, dim=2)         # Shape: (B, T, E, H)

        # Apply gate weights and sum across experts
        gate_weights = gate_weights.unsqueeze(-1)                   # (B, T, E, 1)
        weighted_output = expert_outputs * gate_weights             # (B, T, E, H)
        output = weighted_output.sum(dim=2)                         # (B, T, H)

        return self.dropout(output), gate_weights.squeeze(-1)      # Output + gate weights


# ### DistilBertLayerWithMoE: MoE-Integrated Encoder Block for DistilBERT

# In[ ]:


class DistilBertLayerWithMoE(torch.nn.Module):
    """
    A transformer block for DistilBERT with a Mixture-of-Experts (MoE) feed-forward layer.

    This replaces the standard FFN with a dynamic expert-routing mechanism while preserving
    DistilBERT’s simplified architecture and pre-normalization scheme.

    Parameters:
        config (DistilBertConfig): Model configuration.
        num_experts (int): Number of experts per MoE layer.
        top_k (int): Number of experts to select per token (sparse gating).
    """
    def __init__(self, config, num_experts=7, top_k=2):
        super().__init__()

        from transformers.models.distilbert.modeling_distilbert import MultiHeadSelfAttention

        # Attention sub-layer
        self.dim = config.dim
        self.hidden_dropout = config.dropout
        self.attention = MultiHeadSelfAttention(config)
        self.attention_dropout = torch.nn.Dropout(config.dropout)
        self.attention_norm = torch.nn.LayerNorm(config.dim, eps=1e-12)

        # Mixture-of-Experts as the FFN replacement
        self.intermediate = MoEFeedForward(
            hidden_dim=config.dim,
            intermediate_dim=config.hidden_dim,
            num_experts=num_experts,
            dropout=config.dropout,
            top_k=top_k,
        )
        self.output_dropout = torch.nn.Dropout(config.dropout)
        self.output_norm = torch.nn.LayerNorm(config.dim, eps=1e-12)

    def initialize_experts_from_ffn(self, pretrained_ffn):
        """
        Copy weights from a pretrained FFN into all MoE experts.
        Adds small Gaussian noise to diversify initialization.
        """
        for expert in self.intermediate.experts:
            expert[0].weight.data.copy_(pretrained_ffn[0].weight.data.clone())
            expert[0].bias.data.copy_(pretrained_ffn[0].bias.data.clone())
            expert[2].weight.data.copy_(pretrained_ffn[2].weight.data.clone())
            expert[2].bias.data.copy_(pretrained_ffn[2].bias.data.clone())

            for param in expert.parameters():
                param.data += 0.01 * torch.randn_like(param)

    def forward(self, hidden_states, attention_mask=None, head_mask=None):
        # Pre-norm → self-attention → dropout → residual
        normed_hidden = self.attention_norm(hidden_states)
        attn_output = self.attention(
            normed_hidden, normed_hidden, normed_hidden,
            attention_mask, head_mask=head_mask
        )[0]
        attn_output = self.attention_dropout(attn_output)
        attn_output = attn_output + hidden_states

        # Pre-norm → MoE FFN → dropout → residual
        normed_attn_output = self.output_norm(attn_output)
        ffn_output, gate_weights = self.intermediate(normed_attn_output)
        ffn_output = self.output_dropout(ffn_output)
        ffn_output = ffn_output + attn_output

        return ffn_output, gate_weights


# ### MoeDistilBertModel: Lightweight DistilBERT with Mixture-of-Experts Layers

# In[ ]:


class MoeDistilBertModel(DistilBertPreTrainedModel):
    def __init__(self, config, num_experts=7, top_k=4):
        super().__init__(config)
        self.config = config

        from transformers.models.distilbert.modeling_distilbert import Embeddings

        self.embeddings = Embeddings(config)

        self.encoder_layers = torch.nn.ModuleList([
            DistilBertLayerWithMoE(config, num_experts=num_experts, top_k=top_k)
            for _ in range(config.n_layers)
        ])

        self.init_weights()

        base_model = DistilBertModel(config)
        base_model.eval()

        for i, moe_layer in enumerate(self.encoder_layers):
            ffn1 = base_model.transformer.layer[i].ffn.lin1
            ffn2 = base_model.transformer.layer[i].ffn.lin2

            pretrained_ffn = torch.nn.Sequential(
                ffn1,
                torch.nn.GELU(),
                ffn2
            )

            moe_layer.initialize_experts_from_ffn(pretrained_ffn)

    def forward(self, input_ids, attention_mask=None):
        embedding_output = self.embeddings(input_ids=input_ids)

        if attention_mask is not None:
            extended_attention_mask = attention_mask[:, None, :]
        else:
            extended_attention_mask = None

        hidden_states = embedding_output
        gate_weights_list = []

        for layer in self.encoder_layers:
            hidden_states, gate_weights = layer(hidden_states, attention_mask=extended_attention_mask)
            gate_weights_list.append(gate_weights)

        pooled_output = hidden_states[:, 0]

        return hidden_states, pooled_output, gate_weights_list


# ### MoeDistilBertScorer: Regression Head with Expert Usage & Entropy Regularization

# In[ ]:


class MoeDistilBertScorer(torch.nn.Module):
    """
    A regression head for MoeDistilBertModel that:
    - Supports optional handcrafted feature fusion
    - Computes auxiliary entropy loss for expert diversity
    - Tracks expert usage for diagnostics

    Parameters:
        base_model (MoeDistilBertModel): MoE-augmented DistilBERT encoder.
        dropout (float): Dropout rate after pooling.
        feature_dim (int): Dimensionality of optional handcrafted features.
    """
    def __init__(self, base_model: MoeDistilBertModel, dropout=0.2, feature_dim=0):
        super().__init__()
        self.encoder = base_model
        self.feature_dim = feature_dim

        input_dim = self.encoder.config.dim + feature_dim  # DistilBERT uses `dim` instead of `hidden_size`

        # Output head for regression
        self.regressor = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 1),
            torch.nn.Dropout(dropout)
        )

    def forward(self, input_ids, attention_mask=None, features=None, labels=None, aux_loss_weight=0.5):
        # Run input through encoder
        hidden_states, pooled_output, gate_weights_list = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Analyze last layer gate weights
        last_gate = gate_weights_list[-1]
        if aux_loss_weight > 0 and last_gate.ndim == 3:
            usage_mask = (last_gate > 0).float()
            usage_count = usage_mask.sum(dim=(0, 1))
            self.expert_usage_counts = usage_count.detach().cpu()

            prob_dist = self.expert_usage_counts / self.expert_usage_counts.sum()
            self.expert_entropy = -(prob_dist * torch.log(prob_dist + 1e-8)).sum().item()

        self.last_gate_weights = gate_weights_list[-1].mean(dim=1).detach().cpu()

        # Concatenate handcrafted features if provided
        if features is not None:
            pooled_output = torch.cat([pooled_output, features], dim=-1)

        # Predict scalar score
        score = self.regressor(pooled_output).squeeze(-1)

        # Compute losses if labels are provided
        loss = None
        aux_loss = None
        if labels is not None:
            loss = torch.nn.functional.mse_loss(score, labels.float())

            if gate_weights_list:
                gate_weights = gate_weights_list[-1]
                mean_gates = gate_weights.mean(dim=(0, 1))
                entropy = -(mean_gates * torch.log(mean_gates + 1e-9)).sum()
                aux_loss = -entropy
                loss = loss + aux_loss_weight * aux_loss

        return {
            "loss": loss,
            "logits": score,
            "hidden_states": hidden_states,
            "aux_loss": aux_loss if labels is not None else None
        }


# ### freeze_distilbert_layers: Selective Layer Unfreezing for Efficient Fine-Tuning

# In[ ]:


def freeze_distilbert_layers(model, num_unfrozen=2):
    """
    Freezes all layers of a DistilBERT-based model except the last `num_unfrozen` encoder layers.

    Useful for:
    - Reducing training time and memory
    - Retaining general-purpose language features in early layers
    - Preventing overfitting on small downstream tasks

    Parameters:
        model (torch.nn.Module): A MoeDistilBertModel or compatible DistilBERT-based model.
        num_unfrozen (int): Number of top encoder layers to keep trainable.
    """
    if num_unfrozen is not None: 
        for param in model.parameters():
            param.requires_grad = False  # Freeze entire model by default
    
        # Unfreeze only the last `num_unfrozen` encoder layers
        if hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
            total_layers = len(model.encoder.layer)
            for i in range(total_layers - num_unfrozen, total_layers):
                for param in model.encoder.layer[i].parameters():
                    param.requires_grad = True

