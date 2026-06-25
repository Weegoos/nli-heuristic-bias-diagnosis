# Heuristic Bias Diagnosis in Natural Language Inference Models

This repository contains the final project for the **Applied Machine Learning Course (Summer 2026)**. The project systematically diagnoses and evaluates hidden syntactic shortcut vulnerabilities in deep pre-trained NLI Transformer models using the controlled **HANS (Heuristic Analysis for NLI Systems)** framework.

---

## 📊 Project Overview
Modern NLP models frequently achieve deceptive high accuracy scores by leveraging superficial statistical shortcuts (e.g., lexical overlap) instead of executing genuine semantic inference. This project exposes those systematic boundaries by conducting an isolated subgroup-level stress test across 3 specific linguistic heuristic failure modes:
1. **Lexical Overlap:** Hypothesis words are a strict subset of premise words.
2. **Subsequence:** The hypothesis is a contiguous chunk within the premise.
3. **Constituent:** The hypothesis forms a complete standalone syntactic phrase in the premise tree.

---

## 📁 Repository Structure
The repository is organized as follows:
* `hans.ipynb` — **Main Execution Environment.** This Jupyter Notebook contains the step-by-step EDA, the Word Overlap baseline execution, raw model inference pipeline, and detailed breakdown tables.
* `hans/` — Core HANS dataset files and evaluation tracking tools.
  * `heuristics_evaluation_set.jsonl` — Raw diagnostic validation subset containing target labels and heuristic types.
  * `heuristics_train_set.jsonl` — Companion training adversarial setup.

---

## 🚀 Quick Start Guide

### 1. Prerequisites & Environment Setup
Before execution, ensure your Python environment is set up and all required processing engines are installed. Open your system terminal or the built-in PyCharm Terminal and execute:

```bash
pip install pandas scikit-learn transformers torch tqdm