#!/usr/bin/env python
# coding: utf-8

# # ELECTRA_utils.py

# ###  Import Core ELECTRA Components and Dependencies

# In[ ]:


from transformers.models.electra.modeling_electra import (
    ElectraSelfAttention,     # ELECTRA's self-attention layer
    ElectraSelfOutput,        # Output of the attention block (residual + norm)
    ElectraEmbeddings         # Embedding layer: token + position + segment
)
from transformers import ElectraModel, ElectraPreTrainedModel  # Full ELECTRA model and the base class for subclassing with custom heads
import numpy as np          
import torch            


# ### Mixture-of-Experts Feed-Forward Block (for ELECTRA)

# In[ ]:


class MoEFeedForward(torch.nn.Module):
    """
    A feed-forward layer enhanced with Mixture-of-Experts (MoE) routing.

    Each input token is dynamically routed to a small subset of expert networks,
    allowing for specialization and computational efficiency.
    """
    def __init__(self, hidden_dim, intermediate_dim, num_experts=7, dropout=0.2, top_k=2):
        super().__init__()

        # Store configuration values
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.num_experts = num_experts
        self.top_k = top_k

        # Define the expert networks: each is a 2-layer MLP with GELU activation
        self.experts = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(hidden_dim, intermediate_dim),
                torch.nn.GELU(),
                torch.nn.Linear(intermediate_dim, hidden_dim)
            ) for _ in range(num_experts)
        ])

        # Gating network to assign expert weights per token
        self.gate = torch.nn.Linear(hidden_dim, num_experts)

        # Dropout after combining expert outputs
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        # Compute logits for expert routing
        gate_logits = self.gate(x)  # shape: (batch_size, seq_len, num_experts)

        # Apply top-k gating if enabled
        if self.top_k > 0 and self.top_k < self.num_experts:
            topk_values, topk_indices = torch.topk(gate_logits, self.top_k, dim=-1)
            mask = torch.full_like(gate_logits, float('-inf'))
            mask.scatter_(-1, topk_indices, topk_values)
            gate_weights = torch.nn.functional.softmax(mask, dim=-1)
        else:
            # Use all experts with dense routing
            gate_weights = torch.nn.functional.softmax(gate_logits, dim=-1)

        # Apply each expert to the input independently
        expert_outputs = [expert(x) for expert in self.experts]  # list of (B, T, H)
        expert_outputs = torch.stack(expert_outputs, dim=2)      # shape: (B, T, E, H)

        # Weight each expert output using the gate
        gate_weights = gate_weights.unsqueeze(-1)                # shape: (B, T, E, 1)
        weighted_output = expert_outputs * gate_weights          # shape: (B, T, E, H)

        # Aggregate across experts and apply dropout
        output = weighted_output.sum(dim=2)                      # shape: (B, T, H)

        return self.dropout(output), gate_weights.squeeze(-1)    # return final output and gating weights


# ### ElectraLayerWithMoE: MoE-Augmented ELECTRA Transformer Block

# In[ ]:


class ElectraLayerWithMoE(torch.nn.Module):
    """
    A modified ELECTRA encoder layer where the standard feed-forward network (FFN)
    is replaced by a Mixture-of-Experts (MoE) block.

    This structure retains the original self-attention mechanism while enabling
    dynamic routing through multiple specialized expert sub-networks.
    """
    def __init__(self, config, num_experts=7, top_k=2):
        super().__init__()

        # Self-attention layer from ELECTRA
        self.attention = ElectraSelfAttention(config)

        # Output layer for attention block (includes residual + norm)
        self.attention_output = ElectraSelfOutput(config)

        # Mixture-of-Experts feed-forward block
        self.intermediate = MoEFeedForward(
            num_experts=num_experts,
            hidden_dim=config.hidden_size,
            dropout=config.hidden_dropout_prob,
            intermediate_dim=config.intermediate_size,
            top_k=top_k
        )

        # Final output transformation and normalization
        self.output_dense = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.output_dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.output_norm = torch.nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def initialize_experts_from_ffn(self, pretrained_ffn):
        """
        Initializes all experts in the MoE block by copying weights from a
        pretrained FFN (Linear → GELU → Linear), then adding small noise for diversity.
        """
        for expert in self.intermediate.experts:
            expert[0].weight.data.copy_(pretrained_ffn[0].weight.data.clone())
            expert[0].bias.data.copy_(pretrained_ffn[0].bias.data.clone())
            expert[2].weight.data.copy_(pretrained_ffn[2].weight.data.clone())
            expert[2].bias.data.copy_(pretrained_ffn[2].bias.data.clone())

            for param in expert.parameters():
                param.data += 0.01 * torch.randn_like(param)  # Add slight noise to diversify experts

    def forward(self, hidden_states, attention_mask=None):
        # Compute attention output
        attention_output = self.attention(hidden_states, attention_mask)[0]
        attention_output = self.attention_output(attention_output, hidden_states)

        # Pass through MoE-enhanced feed-forward block
        moe_output, gate_weights = self.intermediate(attention_output)

        # Final residual connection + dropout + layer norm
        ffn_output = self.output_dense(moe_output)
        ffn_output = self.output_dropout(ffn_output)
        layer_output = self.output_norm(ffn_output + attention_output)

        return layer_output, gate_weights


# ### MoeElectraModel: ELECTRA Encoder with Mixture-of-Experts Layers

# In[ ]:


class MoeElectraModel(ElectraPreTrainedModel):
    """
    A customized ELECTRA model that replaces standard FFN blocks in each encoder layer
    with Mixture-of-Experts (MoE) layers.

    Each MoE layer supports top-k expert selection and optionally inherits weights from a
    pretrained ELECTRA model for smoother initialization.

    Parameters:
        config (ElectraConfig): Model configuration object.
        num_experts (int): Number of parallel experts per MoE layer.
        top_k (int): Number of experts to route each token through.
    """
    def __init__(self, config, num_experts=7, top_k=4):
        super().__init__(config)
        self.config = config

        # Embedding layer (token + position + segment)
        self.embeddings = ElectraEmbeddings(config)

        # Stack of encoder layers with MoE FFNs
        self.encoder_layers = torch.nn.ModuleList([
            ElectraLayerWithMoE(config, num_experts=num_experts, top_k=top_k)
            for _ in range(config.num_hidden_layers)
        ])

        # Initialize weights (e.g., for embeddings and layer modules)
        self.init_weights()

        # Load pretrained ELECTRA model to initialize expert FFNs
        base_electra = ElectraModel(config)
        base_electra.eval()

        for i, moe_layer in enumerate(self.encoder_layers):
            intermediate = base_electra.encoder.layer[i].intermediate.dense
            output = base_electra.encoder.layer[i].output.dense

            pretrained_ffn = torch.nn.Sequential(
                intermediate,
                torch.nn.GELU(),
                output
            )

            # Transfer pretrained weights into MoE experts
            moe_layer.initialize_experts_from_ffn(pretrained_ffn)

    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass through MoE-based ELECTRA encoder.

        Parameters:
            input_ids (Tensor): Token IDs (batch, seq_len).
            attention_mask (Tensor, optional): Attention mask for padded tokens.

        Returns:
            Tuple[
                hidden_states (Tensor): Output of final layer (batch, seq_len, hidden),
                pooled_output (Tensor): First-token summary (batch, hidden),
                gate_weights_list (List[Tensor]): Expert weights from each MoE layer
            ]
        """
        embedding_output = self.embeddings(input_ids=input_ids)

        # Convert standard attention mask to extended form
        if attention_mask is not None:
            extended_attention_mask = attention_mask[:, None, None, :]
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        else:
            extended_attention_mask = None

        hidden_states = embedding_output
        gate_weights_list = []

        # Forward pass through all encoder layers
        for layer in self.encoder_layers:
            hidden_states, gate_weights = layer(hidden_states, attention_mask=extended_attention_mask)
            gate_weights_list.append(gate_weights)

        # Use the first token ([CLS] equivalent) as pooled representation
        pooled_output = hidden_states[:, 0]

        return hidden_states, pooled_output, gate_weights_list


# ### MoeElectraScorer: Scoring Head with MoE Expert Diagnostics & Auxiliary Entropy Loss

# In[ ]:


class MoeElectraScorer(torch.nn.Module):
    """
    A scoring module for MoeElectraModel that combines:
    - Final pooled output with optional handcrafted features
    - A regression head for scalar prediction
    - Expert usage diagnostics and entropy-based regularization

    Parameters:
        base_model (MoeElectraModel): The MoE-enabled ELECTRA encoder.
        dropout (float): Dropout rate for the regression layer.
        feature_dim (int): Dimensionality of optional handcrafted features.
    """
    def __init__(self, base_model: MoeElectraModel, dropout=0.2, feature_dim=0):
        super().__init__()
        self.encoder = base_model
        self.feature_dim = feature_dim

        input_dim = self.encoder.config.hidden_size + feature_dim  # Total input to regression head

        self.regressor = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 1),
            torch.nn.Dropout(dropout)
        )

    def forward(self, input_ids, attention_mask=None, features=None, labels=None, aux_loss_weight=0.5):
        """
        Forward pass through the MoE-based scoring head.

        Parameters:
            input_ids (Tensor): Input token IDs.
            attention_mask (Tensor): Binary attention mask.
            features (Tensor, optional): Additional handcrafted features.
            labels (Tensor, optional): Ground truth regression targets.
            aux_loss_weight (float): Weight for the auxiliary entropy loss.

        Returns:
            dict: {
                'loss': total loss (MSE + auxiliary),
                'logits': predicted scores,
                'hidden_states': last encoder outputs,
                'aux_loss': entropy penalty if computed
            }
        """
        hidden_states, pooled_output, gate_weights_list = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        last_gate = gate_weights_list[-1]

        # Track expert usage and entropy if enabled
        if aux_loss_weight > 0 and last_gate.ndim == 3:
            usage_mask = (last_gate > 0).float()
            usage_count = usage_mask.sum(dim=(0, 1))  # Total uses per expert
            self.expert_usage_counts = usage_count.detach().cpu()

            prob_dist = self.expert_usage_counts / self.expert_usage_counts.sum()
            self.expert_entropy = -(prob_dist * torch.log(prob_dist + 1e-8)).sum().item()

        # Save mean gate weights per token (for logging/analysis)
        self.last_gate_weights = gate_weights_list[-1].mean(dim=1).detach().cpu()

        # Append handcrafted features to pooled representation if available
        if features is not None:
            pooled_output = torch.cat([pooled_output, features], dim=-1)

        # Predict score
        score = self.regressor(pooled_output).squeeze(-1)

        loss = None

        if labels is not None:
            # Standard regression loss
            loss = torch.nn.functional.mse_loss(score, labels.float())

            # Add auxiliary entropy regularization
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


# ### freeze_electra_layers: Selective Unfreezing for ELECTRA Fine-Tuning

# In[ ]:


def freeze_electra_layers(model, num_unfrozen=2):
    """
    Freezes all layers of an ELECTRA-based model except for the top `num_unfrozen` encoder layers.

    This is commonly used to:
    - Speed up training
    - Preserve general representations in early layers
    - Reduce risk of overfitting

    Parameters:
        model (torch.nn.Module): A MoeElectraModel or similar ELECTRA-based model.
        num_unfrozen (int): Number of top encoder layers to unfreeze.
    """
    for param in model.parameters():
        param.requires_grad = False  # Freeze all model parameters by default

    # Unfreeze only the top `num_unfrozen` encoder layers if available
    if hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
        total_layers = len(model.encoder.layer)
        for i in range(total_layers - num_unfrozen, total_layers):
            for param in model.encoder.layer[i].parameters():
                param.requires_grad = True

