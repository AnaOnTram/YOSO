# YOSO

A RAG solution for better utilization of model context length and better recall performance.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/AnaOnTram/YOSO.git
   ```
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
1. Edit Config
    ```bash
    nano config.py
    # Look for LLM_CONFIG session
    #change "backend_type": "Model Engine you currently serve"
    #Save and exit
    ```

2. Start the conversation
    ```bash
    python cli.py
    ```
