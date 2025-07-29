#!/usr/bin/env python
# coding: utf-8

# ### DeBERTa_utils.py

# ### Import Core DeBERTa Components and Dependencies

# In[ ]:


from transformers.models.deberta.modeling_deberta import (
    DebertaAttention,        # DeBERTa's self-attention mechanism
    DebertaSelfOutput,       # Output layer after attention with residual connection and layer norm
    DebertaEmbeddings        # Embedding layer (token + position + segment)
)
from transformers import DebertaModel, DebertaPreTrainedModel  # Full DeBERTa model and base class for extending it
import numpy as np         
import torch              


# ### MoEFeedForward: Mixture-of-Experts Feed-Forward Layer (for DeBERTa)

# In[ ]:


class MoEFeedForward(torch.nn.Module):
    """
    Implements a Mixture-of-Experts (MoE) feed-forward layer with top-k gating.

    Parameters:
        hidden_dim (int): Input and output dimensionality.
        intermediate_dim (int): Hidden layer size for each expert.
        num_experts (int): Number of parallel expert networks.
        dropout (float): Dropout rate applied to final output.
        top_k (int): Number of experts selected per input position (top-k gating).
    """
    def __init__(self, hidden_dim, intermediate_dim, num_experts=7, dropout=0.2, top_k=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.num_experts = num_experts
        self.top_k = top_k  

        # Define the expert networks: each is a feed-forward block (Linear → GELU → Linear)
        self.experts = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(hidden_dim, intermediate_dim),
                torch.nn.GELU(),
                torch.nn.Linear(intermediate_dim, hidden_dim)
            ) for expert in range(num_experts)
        ])

        # Gating layer: learns how to route input to experts
        self.gate = torch.nn.Linear(hidden_dim, num_experts)

        # Dropout applied after combining expert outputs
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        gate_logits = self.gate(x)  # (batch, seq_len, num_experts)

        # Apply top-k gating if specified
        if self.top_k > 0 and self.top_k < self.num_experts:
            topk_values, topk_indices = torch.topk(gate_logits, self.top_k, dim=-1)  # Top-k scores and indices
            mask = torch.full_like(gate_logits, float('-inf'))                       # Initialize full mask
            mask.scatter_(-1, topk_indices, topk_values)                             # Fill top-k positions
            gate_weights = torch.nn.functional.softmax(mask, dim=-1)                # Normalize selected gates
        else:
            gate_weights = torch.nn.functional.softmax(gate_logits, dim=-1)         # Use all experts if no top-k

        expert_outputs = [expert(x) for expert in self.experts]     # Apply each expert: list of tensors
        expert_outputs = torch.stack(expert_outputs, dim=2)         # Shape: (batch, seq_len, num_experts, hidden)

        gate_weights = gate_weights.unsqueeze(-1)                   # Shape: (batch, seq_len, num_experts, 1)
        weighted_output = expert_outputs * gate_weights             # Weighted sum over experts
        output = weighted_output.sum(dim=2)                         # Final output after combining experts

        return self.dropout(output), gate_weights.squeeze(-1)       # Return output and gate weights


# ### DebertaLayerWithMoE: DeBERTa Layer with MoE-Enhanced Feed-Forward

# In[ ]:


class DebertaLayerWithMoE(torch.nn.Module):
    """
    A custom DeBERTa encoder layer that replaces the standard FFN block with a 
    Mixture-of-Experts (MoE) module. Keeps the original attention block unchanged.

    Parameters:
        config (DebertaConfig): Model configuration object.
        num_experts (int): Number of parallel experts in the MoE layer.
        top_k (int): Number of experts to select per token (for sparse routing).
    """
    def __init__(self, config, num_experts=7, top_k=2):
        super().__init__()

        self.attention = DebertaAttention(config)         # Original self-attention mechanism
        self.attention_output = DebertaSelfOutput(config) # Output processing after attention

        # Replace standard FFN with a MoE block
        self.intermediate = MoEFeedForward(  
            num_experts=num_experts,
            hidden_dim=config.hidden_size,
            dropout=config.hidden_dropout_prob,
            intermediate_dim=config.intermediate_size,
            top_k=top_k  
        )
        
        # Final projection and residual normalization
        self.output_dense = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.output_dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.output_norm = torch.nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def initialize_experts_from_ffn(self, pretrained_ffn):
        """
        Initialize all MoE experts using weights from a standard FFN.

        Parameters:
            pretrained_ffn (Sequential): A Linear → GELU → Linear module from pretrained DeBERTa.
        """
        for expert in self.intermediate.experts:
            # Copy weights from the pretrained FFN layers
            expert[0].weight.data.copy_(pretrained_ffn[0].weight.data.clone())
            expert[0].bias.data.copy_(pretrained_ffn[0].bias.data.clone())
            expert[2].weight.data.copy_(pretrained_ffn[2].weight.data.clone())
            expert[2].bias.data.copy_(pretrained_ffn[2].bias.data.clone())

            # Add small noise to each expert to promote diversity
            for param in expert.parameters():
                param.data += 0.01 * torch.randn_like(param)

    def forward(self, hidden_states, attention_mask=None):
        """
        Forward pass through the layer.

        Parameters:
            hidden_states (Tensor): Input embeddings (batch_size, seq_len, hidden_dim)
            attention_mask (Tensor, optional): Attention mask to ignore padding tokens.

        Returns:
            Tuple[Tensor, Tensor]:
                - Output embeddings after attention + MoE + normalization
                - Gating weights from MoE layer
        """
        attention_output = self.attention(hidden_states, attention_mask)[0]           # Self-attention block
        attention_output = self.attention_output(attention_output, hidden_states)     # Residual + norm

        moe_output, gate_weights = self.intermediate(attention_output)                # Pass through MoE

        ffn_output = self.output_dense(moe_output)            # Final projection layer
        ffn_output = self.output_dropout(ffn_output)
        layer_output = self.output_norm(ffn_output + attention_output)  # Residual connection + norm

        return layer_output, gate_weights


# ### MoeDebertaModel: Full DeBERTa Encoder with MoE Layers

# In[ ]:


class MoeDebertaModel(DebertaPreTrainedModel):
    """
    DeBERTa model enhanced with Mixture-of-Experts (MoE) layers.

    Each encoder layer uses a MoE-based feed-forward block instead of the standard dense FFN,
    with pretrained weights initialized from a base DeBERTa model.

    Parameters:
        config (DebertaConfig): Model configuration.
        num_experts (int): Number of experts per MoE layer.
        top_k (int): Number of experts selected per token (top-k gating).
    """
    def __init__(self, config, num_experts=7, top_k=4):
        super().__init__(config)
        self.config = config

        self.embeddings = DebertaEmbeddings(config)  # Token, position, and segment embeddings

        # Replace standard encoder with MoE-enhanced layers
        self.encoder_layers = torch.nn.ModuleList([
            DebertaLayerWithMoE(config, num_experts=num_experts, top_k=top_k)
            for _ in range(config.num_hidden_layers)
        ])

        self.init_weights()  # Initialize model weights

        # Load pretrained DeBERTa model for weight transfer
        base_deberta = DebertaModel(config)
        base_deberta.eval()

        # Transfer pretrained FFN weights to each MoE layer's experts
        for i, moe_layer in enumerate(self.encoder_layers):
            intermediate = base_deberta.encoder.layer[i].intermediate.dense
            output = base_deberta.encoder.layer[i].output.dense

            pretrained_ffn = torch.nn.Sequential(
                intermediate,
                torch.nn.GELU(),
                output
            )

            moe_layer.initialize_experts_from_ffn(pretrained_ffn)

    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass through MoE-enhanced DeBERTa model.

        Parameters:
            input_ids (Tensor): Token IDs (batch_size, seq_len).
            attention_mask (Tensor, optional): Binary mask for padded tokens.

        Returns:
            Tuple[Tensor, Tensor, List[Tensor]]:
                - Hidden states (batch, seq_len, hidden_dim)
                - Pooled [CLS]-like output (batch, hidden_dim)
                - List of gate weights from all layers
        """
        embedding_output = self.embeddings(input_ids=input_ids)

        # Convert attention mask to DeBERTa's extended format
        if attention_mask is not None:
            extended_attention_mask = attention_mask[:, None, None, :]
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        else:
            extended_attention_mask = None

        hidden_states = embedding_output
        gate_weights_list = []  # Collect gating info from each MoE layer

        # Forward pass through all MoE-enhanced encoder layers
        for layer in self.encoder_layers:
            hidden_states, gate_weights = layer(hidden_states, attention_mask=extended_attention_mask)
            gate_weights_list.append(gate_weights)

        pooled_output = hidden_states[:, 0]  # Use first token (like [CLS]) as pooled output

        return hidden_states, pooled_output, gate_weights_list


# ### MoeDebertaScorer: Regression Head with Optional Features and MoE Regularization

# In[ ]:


class MoeDebertaScorer(torch.nn.Module):
    """
    A regression model built on top of MoeDebertaModel for scoring tasks.

    It supports:
    - Optional handcrafted features
    - Expert usage diagnostics
    - Entropy-based auxiliary loss to encourage diverse expert routing

    Parameters:
        base_model (MoeDebertaModel): MoE-enhanced DeBERTa encoder.
        dropout (float): Dropout rate for the regression head.
        feature_dim (int): Size of additional handcrafted feature vector (if used).
    """
    def __init__(self, base_model: MoeDebertaModel, dropout=0.2, feature_dim=0):
        super().__init__()
        self.encoder = base_model                    # MoE-enhanced DeBERTa encoder
        self.feature_dim = feature_dim               # Optional external feature dimensionality

        input_dim = self.encoder.config.hidden_size + feature_dim  # Total input to regressor

        # Regression head: outputs a single scalar score
        self.regressor = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 1),
            torch.nn.Dropout(dropout)
        )

    def forward(self, input_ids, attention_mask=None, features=None, labels=None, aux_loss_weight=0.5):
        """
        Forward pass through the scoring model.

        Parameters:
            input_ids (Tensor): Token IDs.
            attention_mask (Tensor): Padding mask.
            features (Tensor, optional): Handcrafted features (same batch size).
            labels (Tensor, optional): Ground truth scores for computing loss.
            aux_loss_weight (float): Weight for entropy-based MoE auxiliary loss.

        Returns:
            dict: {
                'loss': MSE + auxiliary loss (if labels provided),
                'logits': Predicted scores,
                'hidden_states': Final transformer output,
                'aux_loss': Auxiliary entropy penalty (if computed)
            }
        """
        # Run input through encoder
        hidden_states, pooled_output, gate_weights_list = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        last_gate = gate_weights_list[-1]  # Final layer gating weights

        # Auxiliary loss: encourage experts to be used more uniformly
        if aux_loss_weight > 0 and last_gate.ndim == 3:
            usage_mask = (last_gate > 0).float()  # Binary mask for expert usage

            usage_count = usage_mask.sum(dim=(0, 1))  # Count expert activations
            self.expert_usage_counts = usage_count.detach().cpu()

            prob_dist = self.expert_usage_counts / self.expert_usage_counts.sum()
            self.expert_entropy = -(prob_dist * torch.log(prob_dist + 1e-8)).sum().item()

        # Save averaged gate weights per expert for analysis
        self.last_gate_weights = gate_weights_list[-1].mean(dim=1).detach().cpu()

        # Concatenate handcrafted features if provided
        if features is not None:
            pooled_output = torch.cat([pooled_output, features], dim=-1)

        # Predict the essay score
        score = self.regressor(pooled_output).squeeze(-1)

        loss = None

        # Compute loss if labels are provided
        if labels is not None:
            loss = torch.nn.functional.mse_loss(score, labels.float())

            if gate_weights_list:
                gate_weights = gate_weights_list[-1]
                mean_gates = gate_weights.mean(dim=(0, 1))  # Average across batch and sequence
                entropy = -(mean_gates * torch.log(mean_gates + 1e-9)).sum()  # Entropy of expert usage
                aux_loss = -entropy

                # Combine MSE loss with auxiliary entropy loss
                loss = loss + aux_loss_weight * aux_loss

        return {
            "loss": loss,
            "logits": score,
            "hidden_states": hidden_states,
            "aux_loss": aux_loss if labels is not None else None
        }


# ### freeze_deberta_layers: Selective Layer Unfreezing for Fine-Tuning

# In[ ]:


def freeze_deberta_layers(model, num_unfrozen=2):
    """
    Freezes all layers of a DeBERTa-based model except the last `num_unfrozen` encoder layers.

    This is commonly used in transfer learning to:
    - Speed up training
    - Prevent overfitting
    - Preserve pretrained knowledge in lower layers

    Parameters:
        model (torch.nn.Module): A DeBERTa or MoE-DeBERTa model.
        num_unfrozen (int): Number of top encoder layers to keep trainable.
    """
    for param in model.parameters():
        param.requires_grad = False  # Freeze everything by default

    # Selectively unfreeze top N layers from the encoder
    if hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
        total_layers = len(model.encoder.layer)
        for i in range(total_layers - num_unfrozen, total_layers):
            for param in model.encoder.layer[i].parameters():
                param.requires_grad = True

