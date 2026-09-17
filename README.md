# Mitigating LLM Overconfidence via Multi-Model Ensemble Disagreement

An advanced inference-time system architecture designed to detect and correct Large Language Model (LLM) overconfidence. This system uses multi-model consensus checking to calculate a statistical calibration score before delivering answers to users.

## Project Overview
Standard LLMs frequently suffer from calibration errors, expressing 100% verbalized confidence in completely incorrect answers. This project implements an **Inference-Time Cross-Model Ensemble Disagreement Pipeline**. 

Instead of trusting a single model's self-evaluation (which is inherently biased), a target query is routed to three distinct architectural models simultaneously. By measuring the semantic variance and mathematical disagreement among the independent outputs, our system accurately measures internal uncertainty and dampens false confidence.

┌─────────────────┐│   User Query    │└────────┬────────┘│┌─────────────────┼─────────────────┐▼                 ▼                 ▼┌─────────────┐   ┌─────────────┐   ┌─────────────┐│   Model A   │   │   Model B   │   │   Model C   ││ (e.g. Llama)│   │(e.g. Mistral)│  │ (e.g. Phi)  │└──────┬──────┘   └──────┬──────┘   └──────┬──────┘│                 │                 │└─────────────────┼─────────────────┘▼┌───────────────────────────────┐│   Semantic Consensus Engine   ││   (Measures Text Variance)   │└──────────────┬────────────────┘│▼┌───────────────────────────────┐│ Calibrated Output Generation  ││  (Answer + Real Confidence)   │└───────────────────────────────┘

## System Architecture & Mechanics
1. **Parallel Querying:** The orchestrator dispatches the input prompt to three open-weight foundational models (e.g., Llama 3, Mistral, Phi-3).
2. **Consensus Evaluation:** Outputs are analyzed by an embedding/cross-evaluation node to check if the models agree on core facts or diverged heavily.
3. **Calibration Mapping:** A mathematical uncertainty index is generated based on text variance. If disagreement is high, the final response dynamically lowers its confidence metrics or outputs an explicit fallback warning.

## Directory Structure
```text
llm-ensemble-calibration/
├── .env                  # API keys and environment variables
├── requirements.txt      # Project dependencies
├── main.py               # Application entrypoint
└── src/
    ├── ensemble.py       # Multi-model inference manager
    ├── consensus.py      # Statistical text disagreement calculation
    └── evaluator.py      # Benchmark evaluation scripts
```

## Getting Started

### 1. Clone the repository
```bash
git clone https://github.com
cd llm-ensemble-calibration
```

### 2. Set up a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configuration
Create a `.env` file in the root directory:
```env
OPENAI_API_KEY=your_openai_key_here
TOGETHER_API_KEY=your_together_ai_or_ollama_endpoint
```

### 5. Run the pipeline
```bash
python main.py --query "What is the specific heat capacity of gold?"
```

## Core Metrics Measured
* **Expected Calibration Error (ECE):** Used to benchmark the ensemble pipeline against baseline standalone models.
* **Semantic Divergence Score:** Measures the mathematical distance between distinct token distributions of model answers.