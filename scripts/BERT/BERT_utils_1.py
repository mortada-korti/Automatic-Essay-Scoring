#!/usr/bin/env python
# coding: utf-8

# # BERT_utils_1.py

# ### Import Core BERT Components and Dependencies

# In[ ]:


from transformers.models.bert.modeling_bert import (
    BertSelfAttention,
    BertSelfOutput,
    BertEmbeddings,
    BertPooler
)
from transformers import BertModel, BertPreTrainedModel
import torch          


# ### Mixture-of-Experts Feed-Forward Layer for BERT

# In[ ]:


class MoEFeedForward(torch.nn.Module):
    def __init__(self, hidden_dim, intermediate_dim, num_experts=7, dropout=0.2, top_k=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = torch.nn.Linear(hidden_dim, num_experts)
        self.dropout = torch.nn.Dropout(dropout)

        self.experts = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(hidden_dim, intermediate_dim),
                torch.nn.GELU(),
                torch.nn.Linear(intermediate_dim, hidden_dim)
            ) for expert in range(num_experts)
        ])

    def forward(self, x, expert_mask=None):
        gate_logits = self.gate(x)

        if 0 < self.top_k < self.num_experts:
            topk_values, topk_indices = torch.topk(gate_logits, self.top_k, dim=-1)
            masked = torch.full_like(gate_logits, float("-inf"))
            masked.scatter_(-1, topk_indices, topk_values)
            masked = masked - masked.amax(dim=-1, keepdim=True) 
            gate_weights = torch.nn.functional.softmax(masked, dim=-1)
        else:
            logits = gate_logits - gate_logits.amax(dim=-1, keepdim=True) 
            gate_weights = torch.nn.functional.softmax(logits, dim=-1)

        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=2)
        output = (expert_outputs * gate_weights.unsqueeze(-1)).sum(dim=2)
        return self.dropout(output), gate_weights


# ### Custom BERT Layer with Mixture-of-Experts (MoE) Feed-Forward

# In[ ]:


class BertLayerWithMoE(torch.nn.Module):
    def __init__(self, config, num_experts=7, top_k=2):
        super().__init__()
        self.attention = BertSelfAttention(config)
        self.attention_output = BertSelfOutput(config)
        self.intermediate = MoEFeedForward(
            num_experts=num_experts,
            hidden_dim=config.hidden_size,
            dropout=config.hidden_dropout_prob,
            intermediate_dim=config.intermediate_size,
            top_k=top_k
        )
        self.output_dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.output_norm = torch.nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def initialize_experts_from_ffn(self, pretrained_ffn):
        for expert in self.intermediate.experts:
            expert[0].weight.data.copy_(pretrained_ffn[0].weight.data.clone())
            expert[0].bias.data.copy_(pretrained_ffn[0].bias.data.clone())
            expert[2].weight.data.copy_(pretrained_ffn[2].weight.data.clone())
            expert[2].bias.data.copy_(pretrained_ffn[2].bias.data.clone())

            for param in expert.parameters():
                param.data.add_(0.01 * torch.randn_like(param))

    def forward(self, hidden_states, attention_mask=None, expert_mask=None):
        attention_output = self.attention(hidden_states, attention_mask)[0]
        attention_output = self.attention_output(attention_output, hidden_states)
        moe_output, gate_weights = self.intermediate(attention_output, expert_mask=expert_mask)
        ffn_output = self.output_dropout(moe_output)
        layer_output = self.output_norm(ffn_output + attention_output)
        return layer_output, gate_weights


# ### MoeBERTModel: Full BERT Encoder with MoE-Enhanced Layers

# In[ ]:


class MoeBERTModel(BertPreTrainedModel):
    def __init__(self, config, num_experts=7, top_k=2, pretrained_name_or_path="bert-base-uncased"):
        super().__init__(config)
        self.config = config
        self.num_experts = num_experts
        self.top_k = top_k

        base_bert = BertModel.from_pretrained(pretrained_name_or_path, config=config)
        base_bert.eval()

        self.embeddings = base_bert.embeddings
        self.pooler = base_bert.pooler

        self.encoder_layers = torch.nn.ModuleList([
            BertLayerWithMoE(config, num_experts=num_experts, top_k=top_k)
            for _ in range(config.num_hidden_layers)
        ])

        with torch.no_grad():
            for i, moe_layer in enumerate(self.encoder_layers):
                base_layer = base_bert.encoder.layer[i]
                moe_layer.attention = base_layer.attention.self
                moe_layer.attention_output = base_layer.attention.output
                moe_layer.output_norm.weight.copy_(base_layer.output.LayerNorm.weight)
                moe_layer.output_norm.bias.copy_(base_layer.output.LayerNorm.bias)

                pretrained_ffn = torch.nn.Sequential(
                    base_layer.intermediate.dense,
                    torch.nn.GELU(),
                    base_layer.output.dense
                )
                moe_layer.initialize_experts_from_ffn(pretrained_ffn)

        self._last_gate_weights = None

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, expert_mask=None):
        device = input_ids.device
        extended_attention_mask = None

        if attention_mask is not None:
            extended_attention_mask = self.get_extended_attention_mask(
                attention_mask, input_ids.shape, device
            )

        if expert_mask is not None:
            expert_mask = expert_mask.to(device)

        hidden_states = self.embeddings(input_ids=input_ids, token_type_ids=token_type_ids)

        gate_weights_list = []
        for layer in self.encoder_layers:
            hidden_states, gate_weights = layer(
                hidden_states,
                attention_mask=extended_attention_mask,
                expert_mask=expert_mask
            )
            gate_weights_list.append(gate_weights)

        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
            summed = (hidden_states * mask).sum(dim=1)
            denom = mask.sum(dim=1).clamp_min(1e-6)
            pooled_output = summed / denom
        else:
            pooled_output = hidden_states[:, 0]  
            
        self._last_gate_weights = gate_weights_list
        return hidden_states, pooled_output, gate_weights_list


# ### MoeBERTScorer: Essay Scoring Head with Optional Handcrafted Features & MoE Regularization

# In[ ]:


class MoeBERTScorer(torch.nn.Module):
    def __init__(self, base_model: MoeBERTModel, dropout=0.2, feature_dim=0):
        super().__init__()
        self.encoder = base_model
        self.feature_dim = feature_dim

        input_dim = self.encoder.config.hidden_size + feature_dim

        self.regressor = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 1),
            torch.nn.Dropout(dropout)
        )

        # for debugging/inspection
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
        hidden_states, pooled_output, gate_weights_list = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            expert_mask=expert_mask
        )
        last_gate = gate_weights_list[-1] 
        gate_sup_loss = None
        if (expert_mask is not None) and (labels is not None):
            target = torch.nn.functional.one_hot(
                expert_mask, num_classes=last_gate.size(-1)
            ).float()  # (B, E)
            if attention_mask is not None:
                am = attention_mask.unsqueeze(-1).to(last_gate.dtype)  
                p_target = (last_gate * target.unsqueeze(1)).sum(-1)  
                p_target = (p_target * am.squeeze(-1)).sum(1) / am.squeeze(-1).sum(1).clamp_min(1e-6) 
            else:
                p_target = (last_gate * target.unsqueeze(1)).sum(-1).mean(1)
            gate_sup_loss = -torch.log(p_target.clamp_min(1e-12)).mean()

        if aux_loss_weight is not None and aux_loss_weight > 0 and last_gate.ndim == 3:
            with torch.no_grad():
                prob_mass = last_gate.sum(dim=(0, 1))  # (E,)
                self.expert_usage_counts = prob_mass.detach().cpu()
                pm_sum = prob_mass.sum()
                if pm_sum > 0:
                    prob_dist = (prob_mass / (pm_sum + 1e-12)).clamp_min(1e-12)
                    self.expert_entropy = float(-(prob_dist * prob_dist.log()).sum().item())
                else:
                    self.expert_entropy = 0.0

        with torch.no_grad():
            self.last_gate_weights = last_gate.mean(dim=1).detach().cpu()

        if features is not None:
            if features.device != pooled_output.device:
                features = features.to(pooled_output.device)
            if features.dtype != pooled_output.dtype:
                features = features.to(pooled_output.dtype)
            pooled_output = torch.cat([pooled_output, features], dim=-1)

        score = self.regressor(pooled_output).squeeze(-1)  # (B,)

        loss = None
        aux_loss = None

        if labels is not None:
            labels = labels.to(score.dtype)
            loss = torch.nn.functional.mse_loss(score, labels)

            if gate_sup_loss is not None:
                loss = loss + 0.2 * gate_sup_loss 

            if gate_weights_list and aux_loss_weight is not None and aux_loss_weight > 0:
                gate_weights = gate_weights_list[-1]  
                mean_gates = gate_weights.mean(dim=(0, 1))  
                mean_gates = mean_gates / (mean_gates.sum() + 1e-12)
                mean_gates = mean_gates.clamp_min(1e-12)
                entropy = -(mean_gates * mean_gates.log()).sum()
                aux_loss = -entropy  
                loss = loss + aux_loss_weight * aux_loss

        return {
            "loss": loss,
            "logits": score,         
            "hidden_states": hidden_states,
            "aux_loss": aux_loss if labels is not None else None
        }


# In[ ]:


def preprocess(example, tokenizer):
    tokens = tokenizer(
        example["essay"],
        truncation=True,
        padding="max_length",
        max_length=512,
    )
    tokens["labels"] = float(example["normalized_score"])
    feature_prefixes = ("len_", "read_", "comp_", "var_", "sent_")
    feat_items = [(k, v) for k, v in example.items() if k.startswith(feature_prefixes)]
    feat_items.sort(key=lambda kv: kv[0])  
    handcrafted_feats = [float(v) for _, v in feat_items]
    tokens["features"] = handcrafted_feats
    tokens["essay_set"] = int(example["essay_set"])
    return tokens


# In[ ]:


def data_collator(features):
    batch = {}
    if "input_ids" in features[0]:
        batch["input_ids"] = torch.as_tensor([f["input_ids"] for f in features], dtype=torch.long)
    if "attention_mask" in features[0]:
        batch["attention_mask"] = torch.as_tensor([f["attention_mask"] for f in features], dtype=torch.long)
    if "token_type_ids" in features[0] and features[0]["token_type_ids"] is not None:
        batch["token_type_ids"] = torch.as_tensor([f["token_type_ids"] for f in features], dtype=torch.long)
    if "labels" in features[0]:
        batch["labels"] = torch.as_tensor([f["labels"] for f in features], dtype=torch.float32)
    if "features" in features[0] and features[0]["features"] is not None:
        first_feats = features[0]["features"]
        if isinstance(first_feats, (list, tuple)) and len(first_feats) > 0:
            batch["features"] = torch.as_tensor([f["features"] for f in features], dtype=torch.float32)
    if "expert_mask" in features[0]:
        mask_vals = [f["expert_mask"] for f in features]
        if all(m is not None for m in mask_vals):
            batch["expert_mask"] = torch.as_tensor(mask_vals, dtype=torch.long)
    return batch


# In[ ]:


def map_essay_set_to_expert(train_sets):
    return {eid: idx for idx, eid in enumerate(sorted(train_sets))}


# In[ ]:


def add_expert_mask(example, expert_map):
    es = example.get("essay_set", None)
    if es in expert_map:
        example["expert_mask"] = expert_map[es]
    else:
        example["expert_mask"] = None
    return example


# ### Freeze BERT Layers Except the Last Few (Fine-Tuning Strategy)

# In[ ]:


def freeze_bert_layers(model, num_unfrozen=2):
    for p in model.parameters():
        p.requires_grad = False
    encoder = getattr(model, "encoder", model)
    if hasattr(encoder, "encoder_layers"):
        total_layers = len(encoder.encoder_layers)
        start = max(0, total_layers - int(num_unfrozen))
        for i in range(start, total_layers):
            for p in encoder.encoder_layers[i].parameters():
                p.requires_grad = True
    if hasattr(encoder, "pooler") and encoder.pooler is not None:
        for p in encoder.pooler.parameters():
            p.requires_grad = True
    if hasattr(model, "regressor") and model.regressor is not None:
        for p in model.regressor.parameters():
            p.requires_grad = True

