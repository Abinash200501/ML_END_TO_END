## SMS Spam Classifier — End-to-End MLOps Project

This project demonstrates a complete MLOps pipeline for a Spam SMS Classification model using Hugging Face Transformers, ZenML, and DVC, with automated CI/CD on GitHub Actions and deployment to AWS (ECR + EC2).

## Overview

This repository showcases how to take a machine learning model from experimentation to production with full automation, reproducibility, and observability.

The model identifies whether an SMS message is spam or not spam using Natural Language Processing (NLP).
The pipeline includes data ingestion, preprocessing, model training, evaluation, and continuous deployment.

## System Architecture

```bash
┌────────────────────────────────────────────────────────────────┐
│                        DATA LAYER                              │
│  • SMS Dataset (ham/spam labeled messages)                     │
│  • AWS S3 (MLflow artifacts + DVC model storage)               │
└────────────────────────────────────────────────────────────────┘
                              ↓

┌────────────────────────────────────────────────────────────────┐
│                   ORCHESTRATION LAYER (ZenML)                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │    Data      │→ |  Training    │→ │ Evaluation   │          │
│  │  Pipeline    │  │  Pipeline    │  │ Pipeline     │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│         Ingestion → Tokenization → Fine-tune BERT              │
└────────────────────────────────────────────────────────────────┘
                              ↓

┌────────────────────────────────────────────────────────────────┐
│              EXPERIMENT TRACKING (MLflow on AWS EC2)           |
│  • Logs: hyperparameters, metrics, artifacts                   │
│  • Tracks: 20+ training runs with different configs            │
│  • Registry: All model versions with metadata                  │
└────────────────────────────────────────────────────────────────┘
                              ↓

┌────────────────────────────────────────────────────────────────┐
│              MODEL VERSIONING (DVC + Git + S3)                 │
│  • Code versioning: Git (training scripts, configs)            │
│  • Model versioning: DVC (large binary model files)            │
│  • Remote storage: S3 bucket for team collaboration            │
└────────────────────────────────────────────────────────────────┘
                              ↓

┌────────────────────────────────────────────────────────────────┐
│                 DEPLOYMENT (FastAPI + Docker)                  │
│  • REST API: POST /predict endpoint                            │
│  • Container: Docker image with all dependencies               │
│  • Auto-fetch: DVC pulls model if not present                  │
│  • Inference: Real-time spam classification                    │
└────────────────────────────────────────────────────────────────┘
                              ↓

┌────────────────────────────────────────────────────────────────┐
│                     CI/CD (GitHub Actions)                     │
│  • Self-hosted runner on AWS                                   │
│  • Automated: testing, building, deployment                    │
│  • IAM permissions: EC2, ECR, S3 access                        │
└────────────────────────────────────────────────────────────────┘



# hello
