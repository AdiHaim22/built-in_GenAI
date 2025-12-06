🏗️ Built-In: A GenAI System for Floorplan Version Compliance
Detecting Structural Changes Between Two Versions of Apartment Floorplans

Authors: Afik Aharon, Adi Haim, Liran Aichnboim

✨ 1. Project Overview

Real-estate buyers often receive multiple versions of the same apartment floorplan.
Minor changes between versions may lead to financial loss, reduced area, or legal implications, yet buyers have no automated way to validate differences.

Built-In provides an AI-driven solution that automatically detects whether two floorplans differ structurally.

🎯 2. Problem Statement

Automatically detect and report meaningful changes between two versions of the same floorplan (1–2 bedroom units).

Inputs

Two floorplan images (Version A + Version B)

Output
Output	Meaning
0	No meaningful change (stylistic differences only)
1	Structural modification detected
Novelty

First buyer-oriented comparison approach.

Uses CV feature extraction, Generative AI, and synthetic data generation.

🔧 3. Models & Methods
Pipeline
       ┌──────────────────────┐
       │  Stage 1: Preprocess │
       │ Inpainting + LoRA    │
       └───────────┬──────────┘
                   ↓
       ┌──────────────────────┐
       │ Stage 2: Synthetic   │
       │   Data Generation    │
       └───────────┬──────────┘
                   ↓
       ┌──────────────────────┐
       │ Stage 3: Feature     │
       │ Extraction (CNN/VIT) │
       └───────────┬──────────┘
                   ↓
       ┌──────────────────────┐
       │ Stage 4: Embedding   │
       │ Comparison + Classif │
       └──────────────────────┘

Techniques Used

Inpainting (diffusion + prompt engineering)

Feature extraction via CNN & ViT encoders

Binary classifier for structural change

Light LoRA domain fine-tuning

Threshold calibration

📊 4. Data Specification & Generation
Base Dataset

FloorPlansV2 (HuggingFace) – filtered subset

Labeling Scheme
Label	Meaning
0	Stylistic differences only
1	Structural modification
Synthetic Data Requirements

1–2 room apartments

Unified image formatting

Paired samples for training

Generated Data Types

No-change pairs (A → A’)

Changed pairs (A → A*)

📈 5. Metrics & KPIs
Goal

Evaluate the model’s ability to detect:

Small, local structural changes

True structural differences vs. stylistic noise

Quality Per Pipeline Step

Inpainting quality

Embedding stability (CNN/VIT)

Classifier performance

Evaluation Protocols

Train/test split

Cross-validation

Paired before/after floorplans

KPIs

Classification accuracy

Sensitivity to small changes

False positive / false negative rate
