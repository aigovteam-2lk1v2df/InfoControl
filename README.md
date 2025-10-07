# InfoControl

A comprehensive information control system for evaluating and managing sensitive information leakage in AI models. This project provides a modular framework for testing guardrails, evaluating model responses, and ensuring proper handling of sensitive data.

## 🚀 Features

- **Modular Architecture**: Clean separation of concerns with dedicated modules for different components
- **Guardrail Framework**: Advanced guardrail system with configurable LLM models
- **Evaluation System**: Comprehensive evaluation metrics for information correctness and sensitive data leakage
- **RAG Integration**: Simple Retrieval-Augmented Generation for information retrieval
- **Experiment Tracking**: LangSmith integration for experiment management and evaluation
- **Multiple Model Support**: Support for various LLM providers (OpenAI, Together, Google)

## 📁 Project Structure

```
InfoControl/
├── src/
│   ├── datasets.ipynb          # Dataset creation and management
│   ├── workflow.ipynb          # Main workflow and experiments
│   ├── evaluation_prompts.py   # Evaluation prompts for model assessment
│   ├── evaluators.py          # Evaluation functions and metrics
│   ├── guardrails.py          # Guardrail framework implementation
│   ├── guardrail_LLM.py       # Guardrail LLM wrapper
│   ├── simple_RAG.py          # RAG implementation
│   └── assets/                # Documentation assets
├── data/                      # Evaluation datasets
│   ├── Financials/           # Financial documents
│   ├── HR/                   # HR documents and resumes
│   └── eval/                 # Evaluation data files
├── experiments/              # Experiment results and configurations
│   ├── GPT-4o-mini/         # GPT-4o-mini experiment results
│   ├── GPT-OSS-20B/         # GPT-OSS-20B experiment results
│   ├── Gemma-3n-E4B/        # Gemma experiment results
│   └── baseline/            # Baseline experiment results
├── pyproject.toml           # Project dependencies
└── .gitignore              # Git ignore patterns
```

## 🛠️ Installation

### Prerequisites

- Python 3.13+
- UV package manager

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/aigovteam-2lk1v2df/InfoControl.git
   cd InfoControl
   ```

2. **Create virtual environment:**
   ```bash
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   uv sync
   ```

4. **Set up environment variables:**
   Create a `.env` file with your API keys:
   ```env
   OPENAI_API_KEY=your_openai_api_key
   LANGCHAIN_API_KEY=your_langsmith_api_key
   TOGETHER_API_KEY=your_together_api_key
   GOOGLE_API_KEY=your_google_api_key
   ```

## 🚀 Quick Start

1. **Launch Jupyter Lab:**
   ```bash
   jupyter lab
   ```

2. **Run the workflow:**
   - Open `src/workflow.ipynb`
   - Execute cells to set up the system
   - Run experiments with different models

3. **Create datasets:**
   - Open `src/datasets.ipynb`
   - Follow the notebook to create and upload datasets to LangSmith

## 📊 Evaluation Metrics

The system provides three main evaluation metrics:

### 1. Correct Denial Evaluator
- Measures whether the model correctly denies access to sensitive information
- Binary evaluation (0 or 1)
- Critical for security compliance

### 2. Information Correctness Evaluator
- Evaluates the accuracy and completeness of model responses
- Continuous scoring (0.0 to 1.0)
- Measures factual accuracy and semantic preservation

### 3. Sensitive Information Leakage Evaluator
- Detects and measures sensitive information leakage
- Continuous scoring (0.0 to 1.0)
- Higher scores indicate more severe leakage

## 🔧 Configuration

### Experiment Configuration
Experiments are configured via JSON files in the `experiments/` directory:

```json
{
  "modelConfig": {
    "Model": "gpt-4o-mini",
    "Temperature": 0.1
  },
  "evalModelConfig": {
    "Model": "gpt-4o-mini",
    "Temperature": 0.0
  },
  "workflow": ["setup", "retrieve", "generate"]
}
```

### Guardrail Configuration
Guardrails can be configured for different scenarios:
- **Alignment**: Model alignment checks
- **Role Checking**: Role-based access control
- **Post-Processing**: Response sanitization

## 📈 Experiment Results

The system tracks experiments across multiple models and configurations:

- **GPT-4o-mini**: Baseline performance metrics
- **GPT-OSS-20B**: Open-source model evaluation
- **Gemma-3n-E4B**: Google's Gemma model testing

Results are stored in CSV format with detailed metrics for analysis.

## 🧪 Usage Examples

### Basic Guardrail Setup
```python
from guardrails import Guardrails, State
from guardrail_LLM import GuardrailLLM

# Initialize guardrail system
guardrail_llm = GuardrailLLM(model="gemma-3n-e4b-it")
guardrails = Guardrails(vector_store, workflow_llm, guardrail_llm)

# Create and run experiment
graph = guardrails.graph_builder(State, experiment_config).compile()
response = graph.invoke({"question": "What is the employee's salary?"})
```

### Evaluation
```python
from evaluators import correct_denial_evaluator, correctness_evaluator, sensitive_info_evaluator

# Run evaluations
denial_score = correct_denial_evaluator(inputs, outputs, reference_outputs, experiment)
correctness_score = correctness_evaluator(inputs, outputs, reference_outputs, experiment)
leakage_score = sensitive_info_evaluator(inputs, outputs, reference_outputs, experiment)
```

## 📚 Documentation

- **Datasets**: See `src/datasets.ipynb` for dataset creation and management
- **Workflow**: See `src/workflow.ipynb` for the main experimental workflow
- **Evaluation**: See `src/evaluation_prompts.py` and `src/evaluators.py` for evaluation details

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- LangSmith for experiment tracking and evaluation
- LangChain for the evaluation framework
- OpenAI, Together, and Google for model access
- The open-source community for various tools and libraries


---

**InfoControl** - Ensuring AI models handle sensitive information responsibly.