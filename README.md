## SMS Spam Classifier — End-to-End MLOps Project

This project demonstrates a complete MLOps pipeline for a Spam SMS Classification model using Hugging Face Transformers, ZenML, and DVC, with automated CI/CD on GitHub Actions and deployment to AWS (ECR + EC2).

## Overview

This repository showcases how to take a machine learning model from experimentation to production with full automation, reproducibility, and observability.

The model identifies whether an SMS message is spam or not spam using Natural Language Processing (NLP).
The pipeline includes data ingestion, preprocessing, model training, evaluation, and continuous deployment.

## System Architecture

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

## Project Workflow

1. Data & Model Versioning (DVC)

    1. Model artifacts and datasets are versioned with DVC.
    2. Remote storage is configured on Amazon S3, enabling efficient tracking and retrieval.

2. Experiment Tracking (MLflow)

    1. MLflow server runs on a dedicated EC2 instance.
    2. Tracks experiments, hyperparameters, and model performance metrics.

3. Pipeline Orchestration (ZenML)

    1. ZenML manages modular pipelines for:
    2. Data loading
    3. Preprocessing
    4. Model training
    5. Evaluation
    6. Model registration and deployment hooks

4. Continuous Integration (CI)

    1. GitHub Actions workflow automates:
    2. Code quality checks
    3. Dependency installation
    4. Model retraining (if required)
    5. Evaluation and accuracy threshold validation
    6. DVC push for model version updates

5. Continuous Deployment (CD)

    1. A self-hosted GitHub runner on AWS EC2 handles deployment.
    2. Docker image is built and pushed to Amazon ECR.
    3. The latest image is pulled and served via a container on EC2 (port 8000).

## Prerequisites (Summary)

1. For detailed setup steps, refer to [Environment Setup and Deployment Guide](./Environment-setup.md)
2. AWS IAM user with required permissions
3. S3 bucket for DVC and MLflow tracking
4. EC2 instances for MLflow and deployment
5. ECR repository for Docker images
6. GitHub secrets configured for CI/CD

## License
This project is licensed under the MIT License — feel free to use, modify, and build upon it.
