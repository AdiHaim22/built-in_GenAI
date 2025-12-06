🏗️ Built-In: GenAI System for Verifying Compliance Between Floorplan Versions
A GenAI-Based System for Detecting Structural Changes Between Floorplan Versions

Authors: Afik Aharon, Adi Haim, Liran Aichnboim

📌 1. Project Overview

Real-estate buyers often receive multiple versions of the same apartment floorplan (e.g., early contractor plans vs. revised plans).
Small, non-transparent changes can introduce financial risk, reduced square footage, or legal issues, yet there is no existing tool allowing buyers to verify changes easily.

Built-In is a computer-vision and generative-AI based system that automatically detects meaningful structural differences between two versions of the same floorplan.

🎯 2. Problem Statement

Automatically detect and report differences between two versions of the same floorplan (1–2 bedroom units).

Inputs

Two floorplan images (Version A, Version B)

Output

0 → No meaningful structural change (stylistic only)

1 → Structural modification detected

Project Novelty

First buyer-oriented floorplan comparison tool.

Combines CV feature extraction, generative synthetic data, diffusion-based inpainting, and Light LoRA fine-tuning.

🔧 3. Pipeline: Models & Methods
Stage 1: Input Preprocessing

Inpainting for generating controlled modifications

Light LoRA fine-tuning for domain adaptation

Stage 2: Synthetic Data Creation

Generate multiple versions of the same apartment plan

Create both unchanged pairs and changed pairs

Stage 3: Feature Extraction

CNN encoders

ViT (Vision Transformer) encoders

Output: embeddings representing layout structure

Stage 4: Comparison & Classification

Compare embeddings between plan versions

Binary classifier predicts structural change (1) or no change (0)

Adjustments & Fine-Tuning

Prompt-controlled inpainting for realistic modifications

Threshold calibration

Classifier optimization

📊 4. Data Specification & Generation
Base Dataset

FloorPlansV2 (HuggingFace) — filtered subset


Labeling

0 → stylistic only

1 → structural change

Synthetic Data Requirements

1–2 room apartments

Unified image formatting

Paired samples for training

Generated Data Types

No-Change Pairs (A and A’)

Changed Pairs (A and modified A*)

📈 5. Metrics & KPIs
How We Evaluate

Ability to detect small, local structural changes

Accuracy vs. human-verified labels

Performance of each individual step:

Inpainting quality

Embedding similarity stability

Classifier accuracy

Protocols

Train/test split

Cross-validation

Paired before/after datasets

KPIs

Classification accuracy

False positives/false negatives

Sensitivity to minimal structural changes
