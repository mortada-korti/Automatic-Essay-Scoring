#!/usr/bin/env python
# coding: utf-8

# # RoBERTa_utils.py

# ### Import Core RoBERTa Components and Dependencies

# In[ ]:


from transformers.models.roberta.modeling_roberta import (
    RobertaSelfAttention,     # RoBERTa's multi-head self-attention layer
    RobertaSelfOutput,        # Output processing after attention (residual + norm)
    RobertaEmbeddings,        # Token + positional + segment embeddings
)
from transformers import RobertaModel, RobertaPreTrainedModel  
import numpy as np        
import torch                


# ### MoEFeedForward: Mixture-of-Experts FFN for RoBERTa

# In[ ]:


class MoEFeedForward(torch.nn.Module):
    """
    A Mixture-of-Experts (MoE) feed-forward layer with optional sparse (top-k) expert selection.

    Parameters:
        hidden_dim (int): Input and output dimensionality.
        intermediate_dim (int): Hidden size of each expert's feed-forward network.
        num_experts (int): Number of expert networks to choose from.
        dropout (float): Dropout probability applied to final output.
        top_k (int): Number of top experts to route each token through (top-k gating).
    """
    def __init__(self, hidden_dim, intermediate_dim, num_experts=7, dropout=0.2, top_k=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.num_experts = num_experts
        self.top_k = top_k  

        # Define expert feed-forward networks: Linear → GELU → Linear
        self.experts = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(hidden_dim, intermediate_dim),
                torch.nn.GELU(),
                torch.nn.Linear(intermediate_dim, hidden_dim)
            ) for expert in range(num_experts)
        ])

        self.gate = torch.nn.Linear(hidden_dim, num_experts)  # Gating network to assign expert weights
        self.dropout = torch.nn.Dropout(dropout)              # Dropout on the final combined output

    def forward(self, x):
        gate_logits = self.gate(x)  # Raw scores for expert selection: (batch, seq_len, num_experts)

        if self.top_k > 0 and self.top_k < self.num_experts:
            # Top-k sparse expert selection
            topk_values, topk_indices = torch.topk(gate_logits, self.top_k, dim=-1)

            # Create a mask with only top-k values retained
            mask = torch.full_like(gate_logits, float('-inf'))
            mask.scatter_(-1, topk_indices, topk_values)

            # Normalize over top-k experts only
            gate_weights = torch.nn.functional.softmax(mask, dim=-1)
        else:
            # Dense routing: softmax over all experts
            gate_weights = torch.nn.functional.softmax(gate_logits, dim=-1)

        # Apply each expert to input x
        expert_outputs = [expert(x) for expert in self.experts]             # List of (batch, seq_len, hidden)
        expert_outputs = torch.stack(expert_outputs, dim=2)                 # Shape: (batch, seq_len, num_experts, hidden)

        gate_weights = gate_weights.unsqueeze(-1)                           # Shape: (batch, seq_len, num_experts, 1)
        weighted_output = expert_outputs * gate_weights                    # Apply weights
        output = weighted_output.sum(dim=2)                                 # Combine experts into one output

        return self.dropout(output), gate_weights.squeeze(-1)              # Final output and gate weights


# ### RobertaLayerWithMoE: MoE-Enhanced Transformer Layer for RoBERTa

# In[ ]:


class RobertaLayerWithMoE(torch.nn.Module):
    """
    A modified RoBERTa encoder layer where the standard feed-forward network (FFN)
    is replaced by a Mixture-of-Experts (MoE) block.

    Parameters:
        config (RobertaConfig): Hugging Face configuration for RoBERTa.
        num_experts (int): Number of parallel experts in the MoE layer.
        top_k (int): Number of top experts to select per token (sparse routing).
    """
    def __init__(self, config, num_experts=7, top_k=2):
        super().__init__()

        self.attention = RobertaSelfAttention(config)           # Multi-head self-attention
        self.attention_output = RobertaSelfOutput(config)       # Output projection + residual + norm

        # Replace the default FFN with Mixture-of-Experts block
        self.intermediate = MoEFeedForward(
            num_experts=num_experts,
            hidden_dim=config.hidden_size,
            dropout=config.hidden_dropout_prob,
            intermediate_dim=config.intermediate_size,
            top_k=top_k
        )

        # Final dense projection and normalization
        self.output_dense = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.output_dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.output_norm = torch.nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def initialize_experts_from_ffn(self, pretrained_ffn):
        """
        Initialize all experts in the MoE block using weights from a standard pretrained FFN.

        Parameters:
            pretrained_ffn (torch.nn.Sequential): A Linear → GELU → Linear module from RoBERTa.
        """
        for expert in self.intermediate.experts:
            # Copy weights from the standard FFN layers
            expert[0].weight.data.copy_(pretrained_ffn[0].weight.data.clone())
            expert[0].bias.data.copy_(pretrained_ffn[0].bias.data.clone())
            expert[2].weight.data.copy_(pretrained_ffn[2].weight.data.clone())
            expert[2].bias.data.copy_(pretrained_ffn[2].bias.data.clone())

            # Add noise to slightly diversify expert parameters
            for param in expert.parameters():
                param.data += 0.01 * torch.randn_like(param)

    def forward(self, hidden_states, attention_mask=None):
        """
        Forward pass through the modified RoBERTa encoder layer.

        Parameters:
            hidden_states (Tensor): Input tensor (batch, seq_len, hidden_dim).
            attention_mask (Tensor, optional): Attention mask for padding tokens.

        Returns:
            Tuple[Tensor, Tensor]:
                - layer_output: Final hidden states after attention + MoE FFN + normalization
                - gate_weights: Gating weights from MoE for each token
        """
        attention_output = self.attention(hidden_states, attention_mask)[0]       # Self-attention
        attention_output = self.attention_output(attention_output, hidden_states) # Residual + norm

        moe_output, gate_weights = self.intermediate(attention_output)            # MoE FFN

        ffn_output = self.output_dense(moe_output)
        ffn_output = self.output_dropout(ffn_output)
        layer_output = self.output_norm(ffn_output + attention_output)            # Final residual + norm

        return layer_output, gate_weights


# ### MoeRobertaModel: Full RoBERTa Encoder with MoE Integration

# In[ ]:


class MoeRobertaModel(RobertaPreTrainedModel):
    """
    Full RoBERTa model with Mixture-of-Experts (MoE) integrated into every encoder layer.

    Replaces the standard feed-forward block in each layer with a MoE module and 
    optionally initializes each expert using pretrained FFN weights.

    Parameters:
        config (RobertaConfig): Configuration object.
        num_experts (int): Number of experts per MoE layer.
        top_k (int): Number of experts selected per token (top-k routing).
    """
    def __init__(self, config, num_experts=7, top_k=4):
        super().__init__(config)
        self.config = config

        self.embeddings = RobertaEmbeddings(config)  # Token + position embeddings

        # Replace standard encoder layers with MoE-enhanced versions
        self.encoder_layers = torch.nn.ModuleList([
            RobertaLayerWithMoE(config, num_experts=num_experts, top_k=top_k)
            for _ in range(config.num_hidden_layers)
        ])

        self.init_weights()  # Initialize all model parameters

        # Load base RoBERTa model to copy pretrained FFN weights into MoE experts
        base_roberta = RobertaModel(config)
        base_roberta.eval()

        for i, moe_layer in enumerate(self.encoder_layers):
            intermediate = base_roberta.encoder.layer[i].intermediate.dense
            output = base_roberta.encoder.layer[i].output.dense

            # Wrap original FFN for compatibility with expert structure
            pretrained_ffn = torch.nn.Sequential(
                intermediate,
                torch.nn.GELU(),
                output
            )

            # Initialize all MoE experts from this base FFN
            moe_layer.initialize_experts_from_ffn(pretrained_ffn)

    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass through the entire MoE-enhanced RoBERTa encoder.

        Parameters:
            input_ids (Tensor): Token IDs (batch_size, seq_len).
            attention_mask (Tensor, optional): Padding mask.

        Returns:
            Tuple[Tensor, Tensor, List[Tensor]]:
                - hidden_states: Final hidden layer outputs
                - pooled_output: First token (like [CLS]) used as global representation
                - gate_weights_list: Gating weights from all MoE layers
        """
        embedding_output = self.embeddings(input_ids=input_ids)

        if attention_mask is not None:
            extended_attention_mask = attention_mask[:, None, None, :]  # (batch, 1, 1, seq_len)
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        else:
            extended_attention_mask = None

        hidden_states = embedding_output
        gate_weights_list = []  # Track gate weights layer by layer

        # Pass through all encoder layers
        for layer in self.encoder_layers:
            hidden_states, gate_weights = layer(hidden_states, attention_mask=extended_attention_mask)
            gate_weights_list.append(gate_weights)

        pooled_output = hidden_states[:, 0]  # Use first token ([CLS]) as summary

        return hidden_states, pooled_output, gate_weights_list


# ###  MoeRobertaScorer: Essay Scoring Head with MoE Entropy Regularization

# In[ ]:


class MoeRobertaScorer(torch.nn.Module):
    """
    Regression head for MoE-enhanced RoBERTa. Supports:
    - Handcrafted feature fusion
    - Entropy-based auxiliary loss for expert regularization
    - Expert usage diagnostics

    Parameters:
        base_model (MoeRobertaModel): MoE-based encoder.
        dropout (float): Dropout rate in the regression head.
        feature_dim (int): Size of optional handcrafted feature vector.
    """
    def __init__(self, base_model: MoeRobertaModel, dropout=0.2, feature_dim=0):
        super().__init__()
        self.encoder = base_model                     # MoE-RoBERTa encoder
        self.feature_dim = feature_dim                # Optional handcrafted feature dimension

        input_dim = self.encoder.config.hidden_size + feature_dim  # Final input size to regressor

        # Regression head: outputs scalar prediction
        self.regressor = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 1),
            torch.nn.Dropout(dropout)
        )

    def forward(self, input_ids, attention_mask=None, features=None, labels=None, aux_loss_weight=0.5):
        """
        Forward pass through the scoring model.

        Parameters:
            input_ids (Tensor): Token IDs.
            attention_mask (Tensor): Binary attention mask (1 = real token).
            features (Tensor, optional): External handcrafted features.
            labels (Tensor, optional): Ground truth scores for regression.
            aux_loss_weight (float): Weight of entropy-based auxiliary loss.

        Returns:
            dict: {
                'loss': total loss (MSE + entropy),
                'logits': predicted scores,
                'hidden_states': final encoder outputs,
                'aux_loss': entropy-based auxiliary term
            }
        """
        hidden_states, pooled_output, gate_weights_list = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        last_gate = gate_weights_list[-1]  # Final layer gate weights

        # Calculate expert usage stats if regularization is enabled
        if aux_loss_weight > 0 and last_gate.ndim == 3:
            usage_mask = (last_gate > 0).float()                   # Binary mask of active experts
            usage_count = usage_mask.sum(dim=(0, 1))               # Expert usage count
            self.expert_usage_counts = usage_count.detach().cpu()

            prob_dist = self.expert_usage_counts / self.expert_usage_counts.sum()  # Normalize
            self.expert_entropy = -(prob_dist * torch.log(prob_dist + 1e-8)).sum().item()  # Entropy

        # Track average gate distribution (for logging/visualization)
        self.last_gate_weights = gate_weights_list[-1].mean(dim=1).detach().cpu()

        # If handcrafted features are used, concatenate with pooled output
        if features is not None:
            pooled_output = torch.cat([pooled_output, features], dim=-1)

        # Predict final score
        score = self.regressor(pooled_output).squeeze(-1)

        loss = None

        if labels is not None:
            loss = torch.nn.functional.mse_loss(score, labels.float())  # Core regression loss

            if gate_weights_list:
                gate_weights = gate_weights_list[-1]
                mean_gates = gate_weights.mean(dim=(0, 1))              # Average over batch + seq
                entropy = -(mean_gates * torch.log(mean_gates + 1e-9)).sum()
                aux_loss = -entropy

                # Combine MSE loss with MoE auxiliary entropy loss
                loss = loss + aux_loss_weight * aux_loss

        return {
            "loss": loss,
            "logits": score,
            "hidden_states": hidden_states,
            "aux_loss": aux_loss if labels is not None else None
        }


# ###  freeze_roberta_layers: Fine-Tuning Only the Top RoBERTa Layers

# In[ ]:


def freeze_roberta_layers(model, num_unfrozen=2):
    """
    Freezes all layers of a RoBERTa-based model except the last `num_unfrozen` encoder layers.

    This is useful in transfer learning scenarios where:
    - You want to retain pretrained knowledge from lower layers
    - You want to reduce computational cost or memory usage
    - You only need high-level adaptation to your downstream task

    Parameters:
        model (torch.nn.Module): The RoBERTa or MoeRoberta model.
        num_unfrozen (int): Number of top encoder layers to unfreeze for training.
    """
    for param in model.parameters():
        param.requires_grad = False  # Freeze all layers by default

    # Unfreeze only the last `num_unfrozen` encoder layers
    if hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
        total_layers = len(model.encoder.layer)
        for i in range(total_layers - num_unfrozen, total_layers):
            for param in model.encoder.layer[i].parameters():
                param.requires_grad = True

