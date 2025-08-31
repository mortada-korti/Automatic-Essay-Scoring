# Automatic Essay Scoring (AES) — Mixture-of-Experts BERT (Cross-Prompt)

> Research code from my Master's internship (Sorbonne University) at Okayama University: building an Automatic Essay Scoring system that **generalizes to unseen prompts** using a **Mixture-of-Experts (MoE)** inside BERT, evaluated on the **ASAP** dataset with **leave-one-prompt-out** cross-validation.

## ✨ Highlights

- **Problem:** Cross-prompt AES — score essays from *unseen* prompts without retraining.
- **Approach:** Insert lightweight **MoE feed-forward experts** inside BERT encoder layers; a **gating network** selects top-k experts per essay; combine with a small **regression head** for the score.
- **Features:** Neural representations **+** a compact set of **hand-crafted linguistic features** (length, readability, lexical/syntactic variation, sentiment) for extra robustness.
- **Evaluation:** **Quadratic Weighted Kappa (QWK)** as the primary metric (Kaggle standard); **MSE** for optimization/diagnostics.
- **Setting:** **Leave-one-prompt-out** (train on 7 ASAP prompts, test on 1 unseen).
- **Results (MoE-BERT):** Best-config **avg QWK = 0.5479** across folds, peaking at **0.6493** on one test fold; fixed config avg **0.4148** (see full table below).

---

## 🗂 Repository Structure

```
Automatic-Essay-Scoring/
├─ config.py # Centralized paths (data, results, etc.)
├─ original_dataset.tsv # Raw ASAP dump (single file format)
├─ data/
│ └─ dataset.tsv # Preprocessed + column-reduced ASAP data
├─ scripts/
│ ├─ utils.py # Preprocessing, features, splits, helpers
│ └─ BERT/
│ ├─ BERT_utils.py # MoE layers (experts, gating), model glue
│ └─ BERT_trainer.py # Training loop for MoE-BERT cross-prompt
├─ notebooks/
│ ├─ data_preprocessing.ipynb # From original ASAP → data/dataset.tsv
│ └─ BERT/
│ ├─ BERT_utils.ipynb # Exploratory MoE/BERT components
│ └─ BERT_trainer.ipynb # Notebook version of the trainer
└─ results/
└─ BERT/
└─ BEST_results.xlsx # Collected best results per fold/config

```

> If you run from a subfolder (e.g., notebooks), `config.py` ensures imports/paths still work.

---

## 📦 Getting Started

### 1) Environment

- **Python**: 3.9–3.11 recommended  
- **GPU**: CUDA-enabled GPU strongly recommended (training uses HF Transformers)

Install dependencies (create/activate your venv/conda first):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121  
pip install transformers datasets scikit-learn pandas numpy textstat textblob spacy tqdm
python -m spacy download en_core_web_sm
```
> Tip: If you hit tokenizer / torch compiler warnings, the trainer sets:

> TOKENIZERS_PARALLELISM=false, TORCHDYNAMO_DISABLE=1, etc., to keep runs stable.

### 2) Data

This repo already includes:

- `original_dataset.tsv` — merged ASAP dump (as provided).

- `data/dataset.tsv` — reduced and normalized version (columns: `essay_id`, `essay_set`, `essay`, `domain1_score`).

If you want to regenerate `data/dataset.tsv`:

1. Start from `original_dataset.tsv`.

2. Run the notebook: `notebooks/data_preprocessing.ipynb`.

3. Outputs go to `data/dataset.tsv`.

## 🧠 Method

### BERT + Mixture-of-Experts (inside encoder FFN)

- Experts: Parallel lightweight feed-forward sublayers per Transformer block.

- Gating: A small linear gate over the [CLS] (or layer hidden state) yields per-expert weights; top-k sparse activation.

- Auxiliary (load-balancing) loss encourages non-collapsed expert usage.

- Regression head: A single FC layer maps the pooled representation (+ optional features) to a normalized score in [0, 1], later denormalized to the prompt’s original scale for QWK.

### Hand-crafted features (compact set)

- Length/structure: #words, #sentences, avg sentence length, avg word length…

- Readability: Flesch–Kincaid, Gunning Fog, Dale–Chall…

- Lexical & syntactic variation: type-token ratio, POS diversity, simple dependency stats…

- Sentiment: polarity, subjectivity.

> Features are standardized per prompt and concatenated with neural representations before the scoring head.

### Cross-Prompt Evaluation

- ASAP has 8 prompts. We run leave-one-prompt-out:

    - Train on 7 prompts (with prompt-specific expert association during training).

    - Test on the held-out unseen prompt.

- Repeat for all 8 folds; report QWK per fold and macro averages.

### Metrics

- QWK (Quadratic Weighted Kappa) — integer scoring agreement (primary).

- MSE — training loss and secondary diagnostic in normalized score space.

## 🚀 Train & Evaluate

### Simple one-shot run (MoE-BERT)

```bash
python scripts/BERT/BERT_trainer.py
```

What it does:

- Loads `data/dataset.tsv`.

- Builds 8 splits (train on 7 prompts, test on 1).

- For each split, tokenizes essays, computes hand-crafted features, and fine-tunes MoeBERT.

- Saves per-fold results under `results/BERT/` (CSV/XLSX as configured in the script).

Key knobs inside `BERT_trainer.py`:

- `num_experts` (e.g., 7)

- `top_k` (e.g., 2 for sparse gating)

- `unfrozen_layers` (BERT fine-tuning depth)

- `aux_loss_weight` (gate entropy / load-balance strength)

- `learning_rate`, epochs, batch_size, dropout

> Check `scripts/BERT/BERT_utils.py` for the MoE layers (`BertLayerWithMoE`, `MoeBERTModel`, `MoeBERTScorer`) and `scripts/utils.py` for preprocessing, feature computation, denormalization, and split generation.

## 📊 Results (summary)
### Best across-fold settings (selected per test fold)

|    Test set |     QWK    |  LR  | Epochs | Aux Loss | Unfrozen Layers |
| ----------: | :--------: | :--: | :----: | :------: | :-------------: |
|           1 |   0.5322   | 5e-5 |    5   |    0.5   |        2        |
|           2 | **0.6493** | 5e-5 |   15   |    0.5   |        2        |
|           3 |   0.5254   | 5e-5 |    7   |    0.5   |        2        |
|           4 |   0.5801   | 5e-5 |    7   |    0.5   |        2        |
|           5 |   0.5939   | 5e-5 |    7   |    0.5   |      **6**      |
|           6 |   0.5323   | 3e-5 |   15   |    0.5   |        2        |
|           7 |   0.3569   | 5e-5 |    5   |  **0.0** |        2        |
|           8 |   0.6127   | 5e-5 |    5   |  **0.0** |        2        |
| **Average** | **0.5479** |   —  |    —   |     —    |        —        |

> Observations: Source-dependent prompts (3–6) are consistently strong; prompt 7 (wide score range, persuasive/narrative) remains the hardest; some folds prefer longer training or deeper unfreezing.

### Fixed configuration (single setting for all folds)

- Representative fixed setting yields avg QWK = 0.4148 across folds.

- Confirms that prompt-wise tuning (still cross-prompt evaluation) helps.

### Plain cross-prompt (no MoE; train single-prompt → test others)

- Training on some prompts generalizes better (e.g., Train on 3 → avg 0.4702), while others generalize poorly (e.g., Train on 8 → avg 0.1721).

- Highlights the variability of cross-prompt transfer and the benefit of MoE.

> Full per-fold and per-config details are in results/BERT/BEST_results.xlsx and CSV artifacts produced by the trainer.

## 🔧 Reproducing the best folds

### Inside `scripts/BERT/BERT_trainer.py`, set:

- `num_experts=7`, `top_k` typically `7` (sparse gating)

- Try the per-fold settings in the table above (LR, epochs, aux weight, unfreezing)

- Batch size commonly `8`, dropout around `0.2`

Then run:

```bash
python scripts/BERT/BERT_trainer.py
```
The script will skip already-completed configs (it checks the results CSV) and append new runs.

📒 Notebooks

- `notebooks/data_preprocessing.ipynb` — prepares `data/dataset.tsv` (column selection, text normalization).

- `notebooks/BERT/BERT_utils.ipynb` — MoE module exploration.

- `notebooks/BERT/BERT_trainer.ipynb` — interactive training walkthrough.

## 🏛️ Dataset (ASAP)

- 12,976 student essays across 8 prompts, diverse genres (narrative, persuasive, expository, source-dependent).

- Each prompt has its own score range; predictions are produced in [0,1] then denormalized back to that range before QWK.

- Essays are noisy, with spelling/grammar variation — realistic for AES.

## 📚 References

- Devlin et al. (2019) BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.

- Jacobs et al. (1991) Adaptive Mixtures of Local Experts.

- Phandi et al. (2015) Flexible Domain Adaptation for AES (feature-based baselines).

## 🙏 Acknowledgments

- Internship host: Okayama University, Takeuchi Laboratory.

- Academic program: Sorbonne University (MSc).

- Thanks to Prof. Koichi Takeuchi (supervisor) and colleagues for guidance and support.

## 📬 Contact

- Author: Sidi Mohammed Mortada Korti

- For questions or collaboration, please email me at `korti_mortada@yahoo.com`.