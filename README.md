# Locomotion LLM Agent

A LLM-based Locomotion Mode Prediction Agent.

## Installation

```sh
git clone https://github.com/ae5n/locomotion-llm-agent.git
cd locomotion-llm-agent
pip install -r requirements.txt
```

Install [Docker](https://docs.docker.com/get-docker/) and start the vector database service:

```sh
docker compose up -d
```

## Evaluation

1. Edit `config/config.py` and set:
   ```python
   IMAGE_DIR = "/path/to/your/images"
   ```
2. Run the evaluation app:
   ```sh
   PYTHONPATH=. streamlit run evaluation/evaluation_app.py
   ```