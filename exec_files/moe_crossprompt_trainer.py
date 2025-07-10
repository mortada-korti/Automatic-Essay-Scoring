#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from transformers import AutoTokenizer, TrainingArguments, Trainer, EarlyStoppingCallback 
from sklearn.metrics import cohen_kappa_score 
from datetime import datetime  
from datasets import Dataset  
import pandas as pd
import warnings  
import sys, os  
warnings.filterwarnings('ignore') 


# In[ ]:


os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Disable parallel tokenizer threads (prevents unwanted warnings)
os.environ["TORCHINDUCTOR_DISABLE"] = "1"       # Disable TorchInductor (experimental compiler, can cause instability)
os.environ["TORCH_COMPILE_DISABLE"] = "1"       # Disable PyTorch 2.0 compile mode (to avoid compatibility issues)
os.environ["TORCHDYNAMO_DISABLE"] = "1"         # Disable TorchDynamo (used in PyTorch graph capture)


# In[ ]:


ROOT_DIR = os.path.abspath(os.path.join(os.getcwd(), ".."))  # Get the parent directory of the current working directory
sys.path.append(ROOT_DIR)  # Add the root directory to Python's import path so we can import custom modules


# In[ ]:


import torch  

# Import directory paths from config (used to load/save data, models, and results)
from config import DATA_DIR, MODEL_DIR, RESULTS_DIR

# Import custom utility functions from your project
from exec_files.utils import (
    generate_cross_prompt_splits,   # Generates train/val/test splits for cross-prompt learning
    get_split,                      # Retrieves data subsets for specific prompts
    compute_metrics_factory,        # Builds evaluation function using QWK metric
    denormalize_score,              # Converts normalized scores back to original scale                
    freeze_bert_layers,             # Freezes all but the top few layers of BERT (partial fine-tuning)
    EssayScoringMoEModel            # Full essay scoring model using a transformer encoder + MoE head
)


# In[ ]:


PRETRAINED_MODEL = "ModernBERT"  # Identifier for the model used in this experiment
MODEL_NAME = "answerdotai/ModernBERT-base"  # Full Hugging Face model name (used for loading tokenizer & weights)
timestamp = datetime.now().strftime("%Y%m%d_%H%M")  # Generate a unique timestamp (e.g., for saving model files)


# In[ ]:


PRETRAINED_MODEL_DIR = os.path.join(MODEL_DIR, f"{PRETRAINED_MODEL}")  # Create full path to save this model's checkpoints
os.makedirs(PRETRAINED_MODEL_DIR, exist_ok=True)  # Create the directory if it doesn't already exist


# In[ ]:


df = pd.read_csv(f"{DATA_DIR}/dataset.tsv", delimiter="\t", encoding='ISO-8859-1')  # Load the full essay dataset as a DataFrame


# In[ ]:


# Use GPU if available, otherwise fallback to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

# Print how many CUDA-compatible GPUs are available
print(f"CUDA device count: {torch.cuda.device_count()}")

# Print the ID and name of the currently selected GPU device
print(f"Current device: {torch.cuda.current_device()} - {torch.cuda.get_device_name(torch.cuda.current_device())}")


# In[ ]:


def preprocess(example):
    '''
    Tokenizes a single essay and adds its normalized score as the label.
    Used to prepare each sample for training or evaluation.
    '''
    tokens = tokenizer(example["essay"], truncation=True, padding="max_length", max_length=512)
    tokens["labels"] = example["normalized_score"]
    return tokens


# In[ ]:


# Generate a list of train/val/test combinations for cross-prompt learning
splits = generate_cross_prompt_splits()

results_summary = []  # Store QWK results for each split

# Loop through each cross-prompt configuration
for i, split in enumerate(splits):
    # Extract train, validation, and test sets based on prompt IDs
    train_df, val_df, test_df = get_split(df, split["train"], split["val"], split["test"]) 
    
    # Name this split (e.g., test3) and create a directory to save its model
    split_suffix = f"test_moe_{split['test']}"
    split_model_dir = os.path.join(PRETRAINED_MODEL_DIR, split_suffix)
    os.makedirs(split_model_dir, exist_ok=True)
    
    # Print the split info
    print("Train Sets:", split["train"])
    print("Validation Set:", split["val"])
    print("Test Set:", split["test"])
    print("-" * 30)
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = EssayScoringMoEModel(MODEL_NAME, num_experts=6)
    
    # Freeze most of BERT, only train top layers (for efficiency & generalization)
    freeze_bert_layers(model.encoder, num_unfrozen=4)
    
    # Convert pandas DataFrames to Hugging Face Datasets
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    test_dataset = Dataset.from_pandas(test_df)
    
    # Tokenize and prepare the data
    train_dataset = train_dataset.map(preprocess)
    val_dataset = val_dataset.map(preprocess)
    test_dataset = test_dataset.map(preprocess)
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=split_model_dir,            # Where to save model checkpoints
        eval_strategy="epoch",                 # Evaluate after each epoch
        save_strategy="epoch",                 # Save model after each epoch
        learning_rate=5e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=20,
        weight_decay=0.01,                     # Helps regularize the model
        warmup_ratio=0.1,                      # Gradually increase LR at start
        logging_dir="./logs",
        logging_strategy="epoch",
        save_total_limit=1,                    # Only keep the best checkpoint
        load_best_model_at_end=True,           # Restore best checkpoint before final evaluation
        metric_for_best_model="qwk",           # Use QWK to decide the best model
        greater_is_better=True
    )

    # Custom: attach alpha to args so model can read it
    training_args.aux_loss_weight = 0.5
    model.encoder.config.aux_loss_weight = training_args.aux_loss_weight

    # Use your custom trainer that supports regression + QWK metric
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_factory(val_df),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]  # Stop early if val performance doesn't improve
    )
    
    # Train the model
    trainer.train()

    # Evaluate on validation set
    val_metrics = trainer.evaluate()
    val_qwk = val_metrics["eval_qwk"]

    # Save current compute_metrics and disable it for raw predictions
    original_compute_metrics = trainer.compute_metrics
    trainer.compute_metrics = None
    
    # Get raw predictions on test set
    predictions = trainer.predict(test_dataset)
    
    # Restore compute_metrics for future evaluations
    trainer.compute_metrics = original_compute_metrics

    # Denormalize predicted and true scores for proper comparison
    preds = predictions.predictions
    labels = predictions.label_ids
    preds = denormalize_score(preds, test_df)
    labels = denormalize_score(labels, test_df)

    # Gate weights (π values) — shape: (num_samples, num_experts)
    gate_weights = model.last_gate_weights.numpy()
    avg_gate_weights = gate_weights.mean(axis=0)  # shape: (num_experts,)

    # Calculate QWK on test set
    test_qwk = cohen_kappa_score(labels, preds, weights="quadratic")
    print(f"Test QWK: {test_qwk:.4f}")
    
    # Save results for this split
    results_summary.append({
        "train_set": split["train"],
        "val_set": split["val"],
        "test_set": split["test"],
        "val_qwk": round(val_qwk, 4),
        "test_qwk": round(test_qwk, 4),
        **{f"pi_expert_{i}": avg_gate_weights[i] for i in range(len(avg_gate_weights))}
    })

    # Save cumulative results to CSV after each split
    summary_df = pd.DataFrame(results_summary)
    summary_df.to_csv(f"{RESULTS_DIR}/test_results_Moe_{PRETRAINED_MODEL}.csv", index=False)

