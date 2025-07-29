#!/usr/bin/env python
# coding: utf-8

# # BERT_utils.py

# ### Import Core BERT Components and Dependencies

# In[ ]:


from transformers.models.bert.modeling_bert import (
    BertSelfAttention,      # The self-attention mechanism used inside BERT layers
    BertSelfOutput,         # Applies a dense layer + residual connection + layer norm after attention
    BertEmbeddings,         # Converts input token IDs to embeddings (token + segment + position)
    BertPooler              # Produces a single pooled output from the final hidden state
)
from transformers import BertModel, BertPreTrainedModel  # Base BERT model and abstract class for custom BERT extensions
import numpy as np     
import torch           


# ### Mixture-of-Experts Feed-Forward Layer for BERT

# In[ ]:


class MoEFeedForward(torch.nn.Module):
    """
    Implements a Mixture-of-Experts (MoE) feed-forward layer that replaces the 
    standard FFN in BERT. Multiple expert networks process the input, and a gating 
    mechanism determines how to combine their outputs.

    Parameters:
        hidden_dim (int): Input and output dimension (usually the BERT hidden size).
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


# ### Custom BERT Layer with Mixture-of-Experts (MoE) Feed-Forward

# In[ ]:


class BertLayerWithMoE(torch.nn.Module):
    """
    A custom BERT layer that replaces the standard feed-forward sublayer with a 
    Mixture-of-Experts module. It preserves the original BERT attention mechanism
    while injecting MoE routing for better flexibility and parameter efficiency.

    Parameters:
        config (BertConfig): Standard configuration object for BERT.
        num_experts (int): Number of expert FFN modules in the MoE block.
        top_k (int): Number of experts to activate per token (sparse routing).
    """
    def __init__(self, config, num_experts=7, top_k=2):
        super().__init__()

        # BERT's original multi-head self-attention mechanism
        self.attention = BertSelfAttention(config)
        self.attention_output = BertSelfOutput(config)

        # Replace the feed-forward network with a MoE module
        self.intermediate = MoEFeedForward(  
            num_experts=num_experts,
            hidden_dim=config.hidden_size,
            dropout=config.hidden_dropout_prob,
            intermediate_dim=config.intermediate_size,
            top_k=top_k
        )

        # Final projection and normalization (as in original BERT layer)
        self.output_dense = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.output_dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.output_norm = torch.nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def initialize_experts_from_ffn(self, pretrained_ffn):
        """
        Initialize all experts in the MoE block by copying weights from a standard 
        pretrained FFN (e.g., from a vanilla BERT model). Adds small noise to break symmetry.

        Parameters:
            pretrained_ffn (torch.nn.Sequential): A standard FFN block with 2 Linear layers and GELU.
        """
        for expert in self.intermediate.experts:
            # Copy weights and biases from the pretrained FFN
            expert[0].weight.data.copy_(pretrained_ffn[0].weight.data.clone())
            expert[0].bias.data.copy_(pretrained_ffn[0].bias.data.clone())
            expert[2].weight.data.copy_(pretrained_ffn[2].weight.data.clone())
            expert[2].bias.data.copy_(pretrained_ffn[2].bias.data.clone())

            # Add small random noise to each expert to diversify them slightly
            for param in expert.parameters():
                param.data += 0.01 * torch.randn_like(param)

    def forward(self, hidden_states, attention_mask=None):
        """
        Forward pass through the MoE-enhanced BERT layer.

        Parameters:
            hidden_states (Tensor): Input embeddings (batch_size, seq_len, hidden_dim)
            attention_mask (Tensor, optional): Mask for attention (used for padding).

        Returns:
            Tuple[Tensor, Tensor]:
                - Final hidden states after attention + MoE + residual + norm
                - Gating weights from MoE (for interpretability)
        """
        # Standard BERT attention
        attention_output = self.attention(hidden_states, attention_mask)[0]
        attention_output = self.attention_output(attention_output, hidden_states)

        # MoE block replaces the original FFN
        moe_output, gate_weights = self.intermediate(attention_output)

        # Final dense layer, dropout, and residual connection with layer norm
        ffn_output = self.output_dense(moe_output)
        ffn_output = self.output_dropout(ffn_output)
        layer_output = self.output_norm(ffn_output + attention_output)

        return layer_output, gate_weights


# ### MoeBERTModel: Full BERT Encoder with MoE-Enhanced Layers

# In[ ]:


class MoeBERTModel(BertPreTrainedModel):
    """
    Full BERT model where each encoder layer uses a Mixture-of-Experts (MoE) feed-forward module
    instead of the standard FFN. Initialized from a pretrained BERT model to inherit knowledge.

    Parameters:
        config (BertConfig): BERT configuration object.
        num_experts (int): Number of experts per layer.
        top_k (int): Number of active experts selected by the gating mechanism per token.
    """
    def __init__(self, config, num_experts=7, top_k=4):
        super().__init__(config)
        self.config = config

        # Token + position + segment embeddings
        self.embeddings = BertEmbeddings(config)

        # Replace original BERT encoder layers with MoE-enhanced layers
        self.encoder_layers = torch.nn.ModuleList([
            BertLayerWithMoE(config, num_experts=num_experts, top_k=top_k)
            for hidden_layer in range(config.num_hidden_layers)
        ])

        # Pooler: creates a single embedding from the [CLS] token
        self.pooler = BertPooler(config)

        # Initialize parameters
        self.init_weights()

        # Load a standard BERT model to transfer pretrained weights into the MoE experts
        base_bert = BertModel(config)
        base_bert.eval()  # We only use it to read weights

        for i, moe_layer in enumerate(self.encoder_layers):
            # Extract the original FFN components from the i-th BERT layer
            intermediate = base_bert.encoder.layer[i].intermediate.dense
            output = base_bert.encoder.layer[i].output.dense

            # Rebuild the FFN module to match the MoE expert structure
            pretrained_ffn = torch.nn.Sequential(
                intermediate,
                torch.nn.GELU(),
                output
            )

            # Initialize each expert in the MoE layer with this pretrained FFN
            moe_layer.initialize_experts_from_ffn(pretrained_ffn)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        """
        Forward pass through the full MoE-enhanced BERT model.

        Parameters:
            input_ids (Tensor): Token IDs (batch_size, seq_len).
            attention_mask (Tensor, optional): Mask to ignore padded tokens.
            token_type_ids (Tensor, optional): Segment IDs (for sentence pairs).

        Returns:
            Tuple[Tensor, Tensor, List[Tensor]]:
                - Last hidden states for all tokens
                - Pooled output (usually from [CLS] token)
                - List of gating weights from all layers (for inspection/analysis)
        """
        # Get embeddings for input tokens
        embedding_output = self.embeddings(input_ids=input_ids, token_type_ids=token_type_ids)

        # Create attention mask in BERT's expected format
        if attention_mask is not None:
            extended_attention_mask = attention_mask[:, None, None, :]  # (batch, 1, 1, seq_len)
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0  # Masked positions get large negative
        else:
            extended_attention_mask = None

        hidden_states = embedding_output
        gate_weights_list = []  # To store MoE gate weights per layer

        # Pass input through each encoder layer (with MoE)
        for layer in self.encoder_layers:
            hidden_states, gate_weights = layer(hidden_states, attention_mask=extended_attention_mask)
            gate_weights_list.append(gate_weights)

        # Final pooled embedding (typically from [CLS] token)
        pooled_output = self.pooler(hidden_states)

        return hidden_states, pooled_output, gate_weights_list


# ### MoeBERTScorer: Essay Scoring Head with Optional Handcrafted Features & MoE Regularization

# In[ ]:


class MoeBERTScorer(torch.nn.Module):
    """
    A regression head built on top of MoeBERTModel to predict essay scores.
    Supports optional handcrafted features and auxiliary loss for encouraging balanced expert usage.

    Parameters:
        base_model (MoeBERTModel): The base encoder model with MoE layers.
        dropout (float): Dropout probability in the regression layer.
        feature_dim (int): Number of external handcrafted features to concatenate with the [CLS] output.
    """
    def __init__(self, base_model: MoeBERTModel, dropout=0.2, feature_dim=0):
        super().__init__()
        self.encoder = base_model                     # MoeBERT encoder
        self.feature_dim = feature_dim                # Dimension of handcrafted features

        # Total input dimension = [CLS] embedding + optional handcrafted features
        input_dim = self.encoder.config.hidden_size + feature_dim

        # Simple regressor: dense layer + dropout to predict a score
        self.regressor = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 1),
            torch.nn.Dropout(dropout)
        )

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                features=None, labels=None, aux_loss_weight=0.5):
        """
        Forward pass through the model.

        Parameters:
            input_ids (Tensor): Token IDs.
            attention_mask (Tensor, optional): Padding mask.
            token_type_ids (Tensor, optional): Segment IDs.
            features (Tensor, optional): Extra handcrafted features.
            labels (Tensor, optional): Ground truth scores.
            aux_loss_weight (float): Weight for entropy-based auxiliary loss on MoE gating.

        Returns:
            dict: Dictionary with:
                - 'loss': Total loss (MSE + optional MoE entropy loss)
                - 'logits': Predicted scores
                - 'hidden_states': Final hidden layer outputs
                - 'aux_loss': Entropy-based auxiliary regularization term (if labels are provided)
        """
        # Encode input through MoE BERT
        hidden_states, pooled_output, gate_weights_list = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        last_gate = gate_weights_list[-1]  # Use last layer's gate weights for auxiliary metrics

        # If entropy regularization is enabled and shape is valid
        if aux_loss_weight > 0 and last_gate.ndim == 3:
            usage_mask = (last_gate > 0).float()  # Binary mask: which experts are used

            usage_count = usage_mask.sum(dim=(0, 1))  # Count how often each expert is selected
            self.expert_usage_counts = usage_count.detach().cpu()

            # Calculate entropy of expert usage distribution
            prob_dist = self.expert_usage_counts / self.expert_usage_counts.sum()
            self.expert_entropy = -(prob_dist * torch.log(prob_dist + 1e-8)).sum().item()

        # Save mean gate weights per expert (for diagnostics)
        self.last_gate_weights = gate_weights_list[-1].mean(dim=1).detach().cpu()

        # Optionally concatenate handcrafted features
        if features is not None:
            pooled_output = torch.cat([pooled_output, features], dim=-1)

        # Predict final score
        score = self.regressor(pooled_output).squeeze(-1)  # (batch_size,)

        loss = None  # Initialize loss

        if labels is not None:
            # Base loss: mean squared error between predicted and true scores
            loss = torch.nn.functional.mse_loss(score, labels.float())

            if gate_weights_list:
                gate_weights = gate_weights_list[-1]                # Use final layer's gates
                mean_gates = gate_weights.mean(dim=(0, 1))          # Average over batch and sequence
                entropy = -(mean_gates * torch.log(mean_gates + 1e-9)).sum()  # Entropy of expert usage
                aux_loss = -entropy                                 # Encourage high entropy (diverse usage)

                # Add auxiliary loss to total loss
                loss = loss + aux_loss_weight * aux_loss

        return {
            "loss": loss,
            "logits": score,
            "hidden_states": hidden_states,
            "aux_loss": aux_loss if labels is not None else None
        }


# ### Freeze BERT Layers Except the Last Few (Fine-Tuning Strategy)

# In[ ]:


def freeze_bert_layers(model, num_unfrozen=2):
    """
    Freezes most of the BERT model parameters to reduce training cost and avoid overfitting,
    except for the last `num_unfrozen` encoder layers and the pooler.

    Parameters:
        model (torch.nn.Module): A BERT-based model (e.g., MoeBERTModel or BertModel).
        num_unfrozen (int): Number of encoder layers (from the top) to keep trainable.
    """
    # Freeze all parameters by default
    for param in model.parameters():
        param.requires_grad = False

    # If model has an encoder with layers, selectively unfreeze the last few layers
    if hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
        total_layers = len(model.encoder.layer)

        # Unfreeze only the last `num_unfrozen` layers
        for i in range(total_layers - num_unfrozen, total_layers):
            for param in model.encoder.layer[i].parameters():
                param.requires_grad = True

    # Also unfreeze the pooler if present (e.g., to fine-tune [CLS] token representation)
    if hasattr(model, "pooler"):  
        for param in model.pooler.parameters():
            param.requires_grad = True

