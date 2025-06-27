# config.py
import os
import sys

# Automatically detect project root (where this file lives)
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))

# Fix Python import path (needed if running from subfolders like notebooks)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Paths to subfolders
EXEC_DIR = os.path.join(ROOT_DIR, "exec_files")
DATA_DIR = os.path.join(ROOT_DIR, "Dataframes")
MODEL_DIR = os.path.join(ROOT_DIR, "saved_models")
NOTEBOOK_DIR = os.path.join(ROOT_DIR, "notebook_files")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")

# Add exec_files for module import
if EXEC_DIR not in sys.path:
    sys.path.append(EXEC_DIR)