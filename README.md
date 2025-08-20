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

## Data Analysis Dashboard

For interactive data analysis and visualization, see the [data-analysis](./data-analysis/) folder.

### Quick Start for Data Analysis

```sh
cd data-analysis
./start_dashboard.sh
```

Or manually:
```sh
cd data-analysis
python server.py
```

Then open your browser to `http://localhost:8000` to access the interactive dashboard.

## Project Structure

```
locomotion-llm-agent/
├── data-analysis/          # Interactive data analysis dashboard
│   ├── index.html         # Dashboard HTML file
│   ├── server.py          # Local HTTP server
│   ├── start_dashboard.sh # Dashboard launcher script
│   ├── README.md          # Dashboard documentation
│   ├── data/              # Symlink to ../data
│   └── images/            # Symlink to ../images
├── data/                  # Locomotion command data
│   └── data.json         # JSON data file
├── images/                # Image files
├── evaluation/            # Evaluation scripts
├── config/                # Configuration files
└── README.md             # This file
```

## License

This project is open source.