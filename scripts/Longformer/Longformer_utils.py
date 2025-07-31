#!/usr/bin/env python
# coding: utf-8

# # Longformer_utils.py

# ### Import Core Longformer Components and Dependencies

# In[ ]:


from transformers.models.longformer.modeling_longformer import (
    LongformerSelfAttention,
    LongformerSelfOutput,
    LongformerEmbeddings,
    LongformerPooler
)
from transformers import LongformerModel, LongformerPreTrainedModel

import numpy as np
import torch    


# ### Mixture-of-Experts Feed-Forward Layer for Longformer

# In[ ]:


class MoEFeedForward(torch.nn.Module):
    """
    Implements a Mixture-of-Experts (MoE) feed-forward layer that replaces the 
    standard FFN in Longformer. Multiple expert networks process the input, and a gating 
    mechanism determines how to combine their outputs.

    Parameters:
        hidden_dim (int): Input and output dimension (usually the Longformer hidden size).
        intermediate_dim (int): Hidden size within each expert's feed-forward network.
        num_experts (int): Total number of parallel expert networks.
        dropout (float): Dropout probability applied to the final output.
        top_k (int): Number of top experts to select (sparse gating); if set to 0, use all experts.
    """
    def __init__(self, hidden_dim, intermediate_dim, num_experts=7, dropout=0.2, top_k=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.num_experts = num_experts
        self.top_k = top_k  

        # Define the expert networks — each one is a small feed-forward block
        self.experts = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(hidden_dim, intermediate_dim),
                torch.nn.GELU(),
                torch.nn.Linear(intermediate_dim, hidden_dim)
            ) for expert in range(num_experts)
        ])

        # Gating network: assigns weights to each expert based on input
        self.gate = torch.nn.Linear(hidden_dim, num_experts)

        # Dropout applied to final output
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass through the MoE layer.

        Parameters:
            x (Tensor): Input tensor of shape (batch_size, seq_len, hidden_dim)

        Returns:
            Tuple[Tensor, Tensor]: 
                - Output tensor of the same shape as input after combining expert outputs.
                - Gating weights tensor (used for interpretability or diagnostics).
        """
        gate_logits = self.gate(x)  # Compute unnormalized scores for each expert

        # Use top-k gating to select a few experts per token
        if self.top_k > 0 and self.top_k < self.num_experts:
            topk_values, topk_indices = torch.topk(gate_logits, self.top_k, dim=-1)  # (batch, seq_len, top_k)

            # Create a full mask and fill in top-k positions with their logits
            mask = torch.full_like(gate_logits, float('-inf')) 
            mask.scatter_(-1, topk_indices, topk_values)

            # Apply softmax to get normalized gating weights (only top-k will be nonzero)
            gate_weights = torch.nn.functional.softmax(mask, dim=-1)
        else:
            # If top_k is 0 or equal to number of experts, use all experts
            gate_weights = torch.nn.functional.softmax(gate_logits, dim=-1)

        # Compute outputs from each expert
        expert_outputs = [expert(x) for expert in self.experts]  # List of (batch, seq_len, hidden_dim)

        # Stack expert outputs: shape becomes (batch, seq_len, num_experts, hidden_dim)
        expert_outputs = torch.stack(expert_outputs, dim=2)

        # Reshape gate weights to align for broadcasting: (batch, seq_len, num_experts, 1)
        gate_weights = gate_weights.unsqueeze(-1)

        # Multiply each expert's output by its corresponding gate weight
        weighted_output = expert_outputs * gate_weights  # (batch, seq_len, num_experts, hidden_dim)

        # Sum over experts to get final output per token
        output = weighted_output.sum(dim=2)  # (batch, seq_len, hidden_dim)

        return self.dropout(output), gate_weights.squeeze(-1)  # Also return gate weights


# ### Custom Longformer Layer with Mixture-of-Experts (MoE) Feed-Forward

# In[ ]:


class LongformerLayerWithMoE(torch.nn.Module):
    """
    A custom Longformer layer that replaces the standard feed-forward sublayer with a 
    Mixture-of-Experts (MoE) module. It keeps the original attention mechanism intact.
    
    Parameters:
        config (LongformerConfig): Configuration object for Longformer.
        num_experts (int): Number of expert FFNs in the MoE block.
        top_k (int): Number of experts to activate per token (sparse routing).
    """
    def __init__(self, config, num_experts=7, top_k=2):
        super().__init__()
        self.attention = LongformerSelfAttention(config, layer_id=0)
        self.attention_output = LongformerSelfOutput(config)

        self.intermediate = MoEFeedForward(
            num_experts=num_experts,
            hidden_dim=config.hidden_size,
            intermediate_dim=config.intermediate_size,
            dropout=config.hidden_dropout_prob,
            top_k=top_k
        )

        self.output_dense = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.output_dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.output_norm = torch.nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def initialize_experts_from_ffn(self, pretrained_ffn):
        """
        Initialize all experts in the MoE block by copying weights from a standard 
        pretrained FFN (e.g., from a vanilla Longformer model). Adds small noise to break symmetry.

        Parameters:
            pretrained_ffn (torch.nn.Sequential): A standard FFN block with 2 Linear layers and GELU.
        """
        for expert in self.intermediate.experts:
            # Copy weights and biases from the pretrained FFN
            expert[0].weight.data.copy_(pretrained_ffn[0].weight.data.clone())
            expert[0].bias.data.copy_(pretrained_ffn[0].bias.data.clone())
            expert[2].weight.data.copy_(pretrained_ffn[2].weight.data.clone())
            expert[2].bias.data.copy_(pretrained_ffn[2].bias.data.clone())

            # Add small random noise to diversify experts
            for param in expert.parameters():
                param.data += 0.01 * torch.randn_like(param)
                
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        output_attentions=False,
        is_index_masked=None,
        is_global_attn=None
    ):
        """
        Forward pass through Longformer layer with MoE-enhanced FFN.
        
        Args:
            hidden_states: (batch_size, seq_len, hidden_dim)
            attention_mask: Attention mask
            output_attentions: Whether to return attention weights
            is_index_masked: Masked positions
            is_global_attn: Global attention positions
            
        Returns:
            (layer_output, gate_weights)
        """
        attention_output = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            is_index_masked=is_index_masked,
            is_global_attn=is_global_attn
        )[0]

        attention_output = self.attention_output(attention_output, hidden_states)

        moe_output, gate_weights = self.intermediate(attention_output)
        ffn_output = self.output_dense(moe_output)
        ffn_output = self.output_dropout(ffn_output)
        layer_output = self.output_norm(ffn_output + attention_output)

        return layer_output, gate_weights


# ### MoeLongformerModel: Full Longformer Encoder with MoE-Enhanced Layers

# In[ ]:


class MoeLongformerModel(LongformerPreTrainedModel):
    """
    Longformer model with Mixture-of-Experts (MoE) feed-forward blocks in each encoder layer.

    Parameters:
        config (LongformerConfig): Longformer configuration object.
        num_experts (int): Number of experts per layer.
        top_k (int): Number of active experts selected per token.
    """
    def __init__(self, config, num_experts=7, top_k=4):
        super().__init__(config)
        self.config = config

        # Token + position + type embeddings
        self.embeddings = LongformerEmbeddings(config)

        # Stack of encoder layers with MoE FFNs
        self.encoder_layers = torch.nn.ModuleList([
            LongformerLayerWithMoE(config, num_experts=num_experts, top_k=top_k)
            for _ in range(config.num_hidden_layers)
        ])

        # Pooler (optional)
        self.pooler = LongformerPooler(config)

        self.init_weights()

        # Load from pretrained Longformer weights to initialize experts
        base_model = LongformerModel(config)
        base_model.eval()

        for i, moe_layer in enumerate(self.encoder_layers):
            intermediate = base_model.encoder.layer[i].intermediate.dense
            output = base_model.encoder.layer[i].output.dense

            pretrained_ffn = torch.nn.Sequential(
                intermediate,
                torch.nn.GELU(),
                output
            )

            moe_layer.initialize_experts_from_ffn(pretrained_ffn)

    def forward(self, input_ids, attention_mask=None, global_attention_mask=None):
        """
        Forward pass through the MoE-enhanced Longformer.

        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len) → 1 for tokens to attend, 0 for padding
            global_attention_mask: (batch, seq_len) → 1 for global attention tokens

        Returns:
            hidden_states, pooled_output, gate_weights_list
        """
        embedding_output = self.embeddings(input_ids=input_ids)

        hidden_states = embedding_output
        gate_weights_list = []

        # Iterate through each MoE encoder layer
        for layer in self.encoder_layers:
            hidden_states, gate_weights = layer(
                hidden_states,
                attention_mask=attention_mask,
                is_index_masked=(attention_mask == 0 if attention_mask is not None else None),
                is_global_attn=global_attention_mask
            )
            gate_weights_list.append(gate_weights)

        # Apply pooling (e.g., [CLS] token or average pooling)
        pooled_output = self.pooler(hidden_states)

        return hidden_states, pooled_output, gate_weights_list


# ### MoeLongformerScorer: Essay Scoring Head with Optional Handcrafted Features & MoE Regularization

# In[ ]:


class MoeLongformerScorer(torch.nn.Module):
    """
    A regression head built on top of MoeLongformerModel to predict essay scores.
    Supports optional handcrafted features and auxiliary loss for encouraging balanced expert usage.

    Parameters:
        base_model (MoeLongformerModel): The Longformer encoder model with MoE layers.
        dropout (float): Dropout probability in the regression layer.
        feature_dim (int): Number of external handcrafted features to concatenate with the [CLS] output.
    """
    def __init__(self, base_model, dropout=0.2, feature_dim=0):
        super().__init__()
        self.encoder = base_model
        self.feature_dim = feature_dim

        input_dim = self.encoder.config.hidden_size + feature_dim

        self.regressor = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 1),
            torch.nn.Dropout(dropout)
        )

    def forward(self, input_ids, attention_mask=None,
                features=None, labels=None, aux_loss_weight=0.5):
        """
        Forward pass through the model.

        Args:
            input_ids (Tensor): Token IDs.
            attention_mask (Tensor, optional): Attention mask.
            features (Tensor, optional): Handcrafted feature vector.
            labels (Tensor, optional): Ground truth scores.
            aux_loss_weight (float): Entropy regularization weight.

        Returns:
            dict with keys: loss, logits, hidden_states, aux_loss
        """
        hidden_states, pooled_output, gate_weights_list = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        last_gate = gate_weights_list[-1]

        if aux_loss_weight > 0 and last_gate.ndim == 3:
            usage_mask = (last_gate > 0).float()
            usage_count = usage_mask.sum(dim=(0, 1))
            self.expert_usage_counts = usage_count.detach().cpu()

            prob_dist = self.expert_usage_counts / self.expert_usage_counts.sum()
            self.expert_entropy = -(prob_dist * torch.log(prob_dist + 1e-8)).sum().item()

        self.last_gate_weights = last_gate.mean(dim=1).detach().cpu()

        if features is not None:
            pooled_output = torch.cat([pooled_output, features], dim=-1)

        score = self.regressor(pooled_output).squeeze(-1)

        loss = None
        if labels is not None:
            loss = torch.nn.functional.mse_loss(score, labels.float())

            if gate_weights_list:
                gate_weights = gate_weights_list[-1]
                mean_gates = gate_weights.mean(dim=(0, 1))
                entropy = -(mean_gates * torch.log(mean_gates + 1e-9)).sum()
                aux_loss = -entropy
                loss = loss + aux_loss_weight * aux_loss
        else:
            aux_loss = None

        return {
            "loss": loss,
            "logits": score,
            "hidden_states": hidden_states,
            "aux_loss": aux_loss
        }


# ### Freeze Longformer Layers Except the Last Few (Fine-Tuning Strategy)

# In[ ]:


def freeze_longformer_layers(model, num_unfrozen=2):
    """
    Freezes most of the Longformer model parameters to reduce training cost and avoid overfitting,
    except for the last `num_unfrozen` encoder layers and the pooler.

    Parameters:
        model (torch.nn.Module): A Longformer-based model (e.g., MoeLongformerModel or LongformerModel).
        num_unfrozen (int): Number of encoder layers (from the top) to keep trainable.
    """
    # Freeze all parameters initially
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze last few encoder layers
    if hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
        total_layers = len(model.encoder.layer)
        for i in range(total_layers - num_unfrozen, total_layers):
            for param in model.encoder.layer[i].parameters():
                param.requires_grad = True

    # Optional: Unfreeze embeddings if you want to fine-tune lower representations
    if hasattr(model, "embeddings"):
        for param in model.embeddings.parameters():
            param.requires_grad = True

    # Optional: Unfreeze pooler if model has one (usually it does)
    if hasattr(model, "pooler"):
        for param in model.pooler.parameters():
            param.requires_grad = True

