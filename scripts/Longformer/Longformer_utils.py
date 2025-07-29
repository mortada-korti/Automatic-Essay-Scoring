#!/usr/bin/env python
# coding: utf-8

# # Longformer_utils.py

# ### Import Core Longformer Components and Dependencies

# In[ ]:


from transformers.models.longformer.modeling_longformer import (
    LongformerSelfAttention,      # Longformer's sliding-window self-attention mechanism
    LongformerSelfOutput,         # Output transformation after self-attention
    LongformerEmbeddings          # Embedding layer (tokens + position + segment)
)
from transformers import LongformerModel, LongformerPreTrainedModel  # Full Longformer model and base class for custom subclasses  
import numpy as np      
import torch            


# ### MoEFeedForward: Mixture-of-Experts FFN for Longformer

# In[ ]:


class MoEFeedForward(torch.nn.Module):
    """
    Implements a Mixture-of-Experts (MoE) feed-forward layer with top-k sparse gating.

    Parameters:
        hidden_dim (int): Dimensionality of the input and output.
        intermediate_dim (int): Size of each expert's hidden layer.
        num_experts (int): Total number of expert networks.
        dropout (float): Dropout probability applied after combining expert outputs.
        top_k (int): Number of top experts selected per token (enables sparse routing).
    """
    def __init__(self, hidden_dim, intermediate_dim, num_experts=7, dropout=0.2, top_k=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.num_experts = num_experts
        self.top_k = top_k  

        # Define the expert feed-forward sub-networks
        self.experts = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(hidden_dim, intermediate_dim),
                torch.nn.GELU(),
                torch.nn.Linear(intermediate_dim, hidden_dim)
            ) for expert in range(num_experts)
        ])

        # Gating layer that determines expert weights
        self.gate = torch.nn.Linear(hidden_dim, num_experts)

        # Dropout applied after the expert-weighted output
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        gate_logits = self.gate(x)  # Shape: (batch, seq_len, num_experts)

        # Sparse top-k gating (only highest scoring experts are used)
        if self.top_k > 0 and self.top_k < self.num_experts:
            topk_values, topk_indices = torch.topk(gate_logits, self.top_k, dim=-1)
            mask = torch.full_like(gate_logits, float('-inf'))
            mask.scatter_(-1, topk_indices, topk_values)
            gate_weights = torch.nn.functional.softmax(mask, dim=-1)
        else:
            gate_weights = torch.nn.functional.softmax(gate_logits, dim=-1)  # Use all experts if no top-k

        # Compute output from each expert
        expert_outputs = [expert(x) for expert in self.experts]  # List of (batch, seq_len, hidden_dim)
        expert_outputs = torch.stack(expert_outputs, dim=2)      # Shape: (batch, seq_len, num_experts, hidden_dim)

        gate_weights = gate_weights.unsqueeze(-1)                # Shape: (batch, seq_len, num_experts, 1)
        weighted_output = expert_outputs * gate_weights          # Apply gating weights to expert outputs
        output = weighted_output.sum(dim=2)                      # Sum across experts

        return self.dropout(output), gate_weights.squeeze(-1)    # Return final output and gating weights


# ### LongformerLayerWithMoE: Custom Longformer Layer with MoE-FFN

# In[ ]:


class LongformerLayerWithMoE(torch.nn.Module):
    """
    A modified Longformer encoder layer that replaces the standard feed-forward network (FFN)
    with a Mixture-of-Experts (MoE) block. Retains Longformer's self-attention mechanism.

    Parameters:
        config (LongformerConfig): Longformer model configuration.
        num_experts (int): Number of experts in the MoE block.
        top_k (int): Number of experts selected per token (enables sparse routing).
    """
    def __init__(self, config, num_experts=7, top_k=2):
        super().__init__()

        self.attention = LongformerSelfAttention(config)          # Sliding window attention
        self.attention_output = LongformerSelfOutput(config)      # Residual + layer norm after attention

        # Replace original FFN with Mixture-of-Experts feed-forward layer
        self.intermediate = MoEFeedForward(
            num_experts=num_experts,
            hidden_dim=config.hidden_size,
            dropout=config.hidden_dropout_prob,
            intermediate_dim=config.intermediate_size,
            top_k=top_k
        )

        # Final transformation and normalization
        self.output_dense = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.output_dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.output_norm = torch.nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def initialize_experts_from_ffn(self, pretrained_ffn):
        """
        Copies weights from a standard FFN into all MoE experts, with added noise to differentiate them.

        Parameters:
            pretrained_ffn (torch.nn.Sequential): Pretrained FFN (Linear → GELU → Linear).
        """
        for expert in self.intermediate.experts:
            expert[0].weight.data.copy_(pretrained_ffn[0].weight.data.clone())
            expert[0].bias.data.copy_(pretrained_ffn[0].bias.data.clone())
            expert[2].weight.data.copy_(pretrained_ffn[2].weight.data.clone())
            expert[2].bias.data.copy_(pretrained_ffn[2].bias.data.clone())

            # Add small noise to each expert's parameters to break symmetry
            for param in expert.parameters():
                param.data += 0.01 * torch.randn_like(param)

    def forward(self, hidden_states, attention_mask=None):
        """
        Forward pass through the Longformer encoder layer with MoE.

        Parameters:
            hidden_states (Tensor): Input embeddings (batch, seq_len, hidden_dim).
            attention_mask (Tensor, optional): Attention mask for sliding-window attention.

        Returns:
            Tuple[Tensor, Tensor]:
                - layer_output: Output after attention + MoE + residual connection
                - gate_weights: Gating distribution from MoE (for interpretability)
        """
        attention_output = self.attention(hidden_states, attention_mask)[0]       # Self-attention step
        attention_output = self.attention_output(attention_output, hidden_states) # Residual + norm

        moe_output, gate_weights = self.intermediate(attention_output)            # MoE FFN block

        ffn_output = self.output_dense(moe_output)
        ffn_output = self.output_dropout(ffn_output)
        layer_output = self.output_norm(ffn_output + attention_output)            # Final residual + norm

        return layer_output, gate_weights


# ###  MoeLongformerModel: Full Longformer Encoder with MoE Layers

# In[ ]:


class MoeLongformerModel(LongformerPreTrainedModel):
    """
    Longformer model enhanced with Mixture-of-Experts (MoE) feed-forward layers.

    This class replaces each standard FFN block with a MoE layer and optionally 
    initializes those experts using weights from a pretrained Longformer model.

    Parameters:
        config (LongformerConfig): Hugging Face configuration object.
        num_experts (int): Number of expert networks in each MoE block.
        top_k (int): Number of experts selected per token (enables sparse routing).
    """
    def __init__(self, config, num_experts=7, top_k=4):
        super().__init__(config)
        self.config = config

        self.embeddings = LongformerEmbeddings(config)  # Token + positional embeddings

        # Replace standard encoder layers with MoE-enhanced ones
        self.encoder_layers = torch.nn.ModuleList([
            LongformerLayerWithMoE(config, num_experts=num_experts, top_k=top_k)
            for hidden_layer in range(config.num_hidden_layers)
        ])

        self.init_weights()  # Initialize model weights

        # Load a pretrained Longformer model for expert initialization
        base_longformer = LongformerModel(config)
        base_longformer.eval()

        for i, moe_layer in enumerate(self.encoder_layers):
            # Extract the FFN sublayers from pretrained Longformer layer
            intermediate = base_longformer.encoder.layer[i].intermediate.dense
            output = base_longformer.encoder.layer[i].output.dense

            # Wrap it into a structure matching the MoE FFN format
            pretrained_ffn = torch.nn.Sequential(
                intermediate,
                torch.nn.GELU(),
                output
            )

            # Initialize each expert with the same pretrained FFN (adds noise internally)
            moe_layer.initialize_experts_from_ffn(pretrained_ffn)

    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass through the MoE-enhanced Longformer.

        Parameters:
            input_ids (Tensor): Input token IDs (batch, seq_len).
            attention_mask (Tensor, optional): Attention mask indicating valid tokens.

        Returns:
            Tuple[Tensor, Tensor, List[Tensor]]:
                - hidden_states: Final hidden states from the encoder
                - pooled_output: [CLS]-like representation from first token
                - gate_weights_list: List of expert gating weights from each layer
        """
        embedding_output = self.embeddings(input_ids=input_ids)

        # Convert to extended attention mask format used by Longformer
        if attention_mask is not None:
            extended_attention_mask = attention_mask[:, None, None, :]
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        else:
            extended_attention_mask = None

        hidden_states = embedding_output
        gate_weights_list = []  # Store gating weights from all layers

        # Pass input through each MoE-enhanced encoder layer
        for layer in self.encoder_layers:
            hidden_states, gate_weights = layer(hidden_states, attention_mask=extended_attention_mask)
            gate_weights_list.append(gate_weights)

        pooled_output = hidden_states[:, 0]  # Use first token (like [CLS]) for pooled output

        return hidden_states, pooled_output, gate_weights_list


# ### MoeLongformerScorer: Essay Scoring Head with MoE Expert Diagnostics & Optional Features

# In[ ]:


class MoeLongformerScorer(torch.nn.Module):
    """
    Regression head for MoeLongformerModel, tailored for tasks like automated essay scoring.

    Supports:
    - Additional handcrafted features
    - Entropy-based auxiliary loss to regularize expert usage
    - Gate usage diagnostics for interpretability

    Parameters:
        base_model (MoeLongformerModel): MoE-enhanced Longformer encoder.
        dropout (float): Dropout rate for regression head.
        feature_dim (int): Size of additional handcrafted features.
    """
    def __init__(self, base_model: MoeLongformerModel, dropout=0.2, feature_dim=0):
        super().__init__()
        self.encoder = base_model                      # Longformer encoder with MoE layers
        self.feature_dim = feature_dim                 # Optional handcrafted feature dimension

        input_dim = self.encoder.config.hidden_size + feature_dim  # Final input size to the regression head

        # Regression head to predict a single scalar (score)
        self.regressor = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 1),
            torch.nn.Dropout(dropout)
        )

    def forward(self, input_ids, attention_mask=None, features=None, labels=None, aux_loss_weight=0.5):
        """
        Forward pass for essay scoring.

        Parameters:
            input_ids (Tensor): Input token IDs.
            attention_mask (Tensor): Attention mask (1 for real tokens, 0 for padding).
            features (Tensor, optional): Handcrafted features to concatenate to [CLS] representation.
            labels (Tensor, optional): True scores for supervised learning.
            aux_loss_weight (float): Weight for auxiliary loss on expert routing entropy.

        Returns:
            dict: {
                'loss': total training loss (MSE + aux),
                'logits': predicted scores,
                'hidden_states': transformer outputs,
                'aux_loss': expert routing entropy penalty (optional)
            }
        """
        # Run inputs through MoE Longformer encoder
        hidden_states, pooled_output, gate_weights_list = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        last_gate = gate_weights_list[-1]  # Gating weights from final encoder layer

        # Track expert usage stats and entropy for auxiliary loss
        if aux_loss_weight > 0 and last_gate.ndim == 3:
            usage_mask = (last_gate > 0).float()  # Mask of activated experts
            usage_count = usage_mask.sum(dim=(0, 1))  # Count of times each expert is used
            self.expert_usage_counts = usage_count.detach().cpu()

            # Compute entropy of expert selection distribution
            prob_dist = self.expert_usage_counts / self.expert_usage_counts.sum()
            self.expert_entropy = -(prob_dist * torch.log(prob_dist + 1e-8)).sum().item()

        # Save averaged gate weights for later visualization
        self.last_gate_weights = gate_weights_list[-1].mean(dim=1).detach().cpu()

        # Concatenate handcrafted features (if available)
        if features is not None:
            pooled_output = torch.cat([pooled_output, features], dim=-1)

        # Predict final score
        score = self.regressor(pooled_output).squeeze(-1)

        loss = None

        # Compute training loss if labels provided
        if labels is not None:
            loss = torch.nn.functional.mse_loss(score, labels.float())  # Main regression loss

            if gate_weights_list:
                gate_weights = gate_weights_list[-1]
                mean_gates = gate_weights.mean(dim=(0, 1))  # Average gate probabilities across batch and time

                entropy = -(mean_gates * torch.log(mean_gates + 1e-9)).sum()  # Entropy penalty
                aux_loss = -entropy

                loss = loss + aux_loss_weight * aux_loss  # Add weighted auxiliary loss

        return {
            "loss": loss,
            "logits": score,
            "hidden_states": hidden_states,
            "aux_loss": aux_loss if labels is not None else None
        }


# ### freeze_longformer_layers: Unfreeze Only Top Layers for Fine-Tuning

# In[ ]:


def freeze_longformer_layers(model, num_unfrozen=2):
    """
    Freezes all layers in a Longformer-based model except for the top `num_unfrozen` encoder layers.

    This helps reduce training cost and overfitting by only tuning the highest (most task-specific) layers.

    Parameters:
        model (torch.nn.Module): A Longformer or MoeLongformer model.
        num_unfrozen (int): Number of top encoder layers to keep trainable.
    """
    for param in model.parameters():
        param.requires_grad = False  # Freeze all parameters by default

    # Unfreeze the last `num_unfrozen` encoder layers if available
    if hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
        total_layers = len(model.encoder.layer)
        for i in range(total_layers - num_unfrozen, total_layers):
            for param in model.encoder.layer[i].parameters():
                param.requires_grad = True

