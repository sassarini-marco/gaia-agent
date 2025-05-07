---
title: Template Final Assignment
emoji: 🕵🏻‍♂️
colorFrom: indigo
colorTo: indigo
sdk: gradio
sdk_version: 5.25.2
app_file: main.py
pinned: false
hf_oauth: true
# optional, default duration is 8 hours/480 minutes. Max duration is 30 days/43200 minutes.
hf_oauth_expiration_minutes: 480
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference




# GAIA Agent

This repository contains an agent system designed to tackle tasks from the GAIA (General AI Assistants) benchmark. It employs a multi-agent architecture, a rich set of tools for web research, file manipulation, and multi-modal understanding, and is configurable to use various large language models.

The system is built using the `smolagents` framework and is structured for both local evaluation and deployment on Hugging Face Spaces.

## ✨ Features

*   **Hierarchical Agent System:**
    *   `ManagerAgent`: Orchestrates tasks, can break them down, delegate to specialized agents, and synthesize results. It's a `GAIACodeAgent` capable of executing Python code.
    *   `WebResearchAgent`: A `GAIAToolCallingAgent` specialized in web searches (Google, DuckDuckGo, Wikipedia), visiting web pages, navigating content, and accessing web archives.
*   **Extensive Toolset:**
    *   **Enhanced Search:** Robust search tools (Google, DuckDuckGo, Wikipedia) with rate limiting and retries.
    *   **Text Web Browser:** Tools to visit URLs, navigate pages (up/down), find text within pages (Ctrl+F like), and search web archives.
    *   **Comprehensive File Conversion (`mdconvert`):** Converts a wide array of file types to Markdown for text-based processing, including:
        *   HTML, PDF, DOCX, XLSX, PPTX
        *   Audio (WAV, MP3, M4A with transcription)
        *   Images (with metadata extraction, and potentially OCR/description if extended)
        *   ZIP archives (extracts and lists contents)
        *   YouTube videos (extracts metadata, description, and transcript)
    *   **Text Inspector:** Reads various local file types (PDF, DOCX, XLSX, audio, etc.) as text and can answer questions about their content using an LLM.
    *   **Visual Question Answering**
    *   **Speech-to-Text:** Converts audio to text.
*   **Multi-Modal Capabilities:** Can process and reason about text, web content, images, audio, and various document formats.
*   **Configurable LLMs:** Utilizes `LiteLLM` for model access, supporting various providers (Gemini, OpenAI, Anthropic, etc.) based on configuration.
*   **GAIA Benchmark Focused:**
    *   Prompts are tailored for GAIA requirements (see `gaia_agent/prompts/`).
    *   Output formatting adheres to GAIA standards.
    *   Includes a local scoring script (`scripts/gaia_scorer.py`).
*   **Local Evaluation:** Script provided to run the agent on GAIA dataset samples locally and score its performance.
*   **Hugging Face Space Integration:**
    *   `main.py` provides a Gradio interface for running evaluations and submitting to the GAIA benchmark leaderboard via a Hugging Face Space.
    *   Supports Hugging Face OAuth for submissions.

## 📂 Project Structure

```
└── gaia-agent/
    ├── README.md                 # This file
    ├── LICENSE                   # MIT License
    ├── main.py                   # Gradio app for Hugging Face Space evaluation
    ├── requirements.txt          # Python dependencies
    ├── gaia_agent/               # Core agent logic
    │   ├── agents/               # Agent implementations (Manager, WebResearch)
    │   ├── config/               # Configuration (settings.py)
    │   ├── prompts/              # YAML prompt templates for agents
    │   ├── tools/                # Various tools used by agents
    │   └── utils/                # Utility functions (agent setup, wrappers)
    └── scripts/                  # Utility scripts
        ├── download_gaia.py      # Script to download GAIA dataset
        └── gaia_scorer.py        # Script for local evaluation and scoring
```

## 🚀 Getting Started

### 1. Prerequisites

*   Python 3.9+
*   Git

### 2. Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository_url> # Replace with the actual URL
    cd gaia-agent
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### 3. Configuration

The agent system relies on API keys and other settings defined in `gaia_agent/config/settings.py`, which are loaded from environment variables or a `.env` file.

1.  **Create a `.env` file** in the project root directory (e.g., `sassarini-marco-gaia-agent/.env`).
2.  **Add necessary API keys and configurations.** Refer to `gaia_agent/config/settings.py` for all possible variables. Key ones include:

    *   **LLM API Keys:**
        *   `GOOGLE_API_KEY`: For Google Gemini models.
        *   `GOOGLE_CLOUD_PROJECT`: Required if using Gemini via Google Vertex AI.
        *   `ANTHROPIC_API_KEY`: For Anthropic Claude models.
        *   `OPENAI_API_KEY`: For OpenAI models (or compatible APIs like Fireworks).
        *   `HUGGINGFACE_TOKEN`: For Hugging Face Hub private models or Inference API.
    *   **Search Tool API Keys:**
        *   `SERPAPI_API_KEY`: For SERPAPI Google Search.
        *   `SERPER_API_KEY`: For Serper.dev Google Search.
    *   **Model Configuration (Optional - defaults are provided):**
        *   `MANAGER_MODEL_ID` (default: `gemini-2.0-flash`)
        *   `WEB_MODEL_ID` (default: `gemini-2.0-flash`)
        *   `TEXT_INSPECTOR_MODEL_ID` (default: `gemini-2.0-flash`)
    *   **Google Application Credentials (for Vertex AI):**
        If you are using Google Vertex AI and have service account JSON credentials, you can set the `GOOGLE_APPLICATION_CREDENTIALS_CONTENT` environment variable to the *content* of the JSON file. The system will handle creating a temporary credentials file.
        ```env
        # Example .env file content
        GOOGLE_API_KEY="your_google_api_key"
        GOOGLE_CLOUD_PROJECT="your_gcp_project_id"
        OPENAI_API_KEY="your_openai_api_key"
        SERPER_API_KEY="your_serper_api_key"
        # ... other keys and settings
        ```

### 4. Download GAIA Dataset (for local evaluation)

The `scripts/download_gaia.py` script can be used to download specific levels and splits of the GAIA dataset.

```bash
python scripts/download_gaia.py --level 1 --split validation --output_dir ./data/gaia
```
This will download GAIA Level 1 validation set to `./data/gaia/gaia_level1_validation.jsonl`.
The default data directory expected by `settings.py` is `gaia_agent/data/gaia`. You might need to adjust paths or move the downloaded files accordingly, or change `gaia_data_dir` in your `.env`.

## 🛠️ Usage

### A. Running on Hugging Face Spaces (for official benchmark submission)

The `main.py` file is designed to be run as a Hugging Face Gradio Space.

1.  Ensure your Space is configured with the necessary environment variables (API keys) as secrets.
2.  The `README.md` at the root of this repository (the one you are reading might be a replacement if you cloned this for local use) should be configured for the HF Space (e.g., `app_file: main.py`).
3.  Once deployed, you can log in with your Hugging Face account via the "Login" button in the Gradio interface.
4.  Click "Run Evaluation & Submit All Answers" to:
    *   Fetch all questions from the GAIA benchmark server.
    *   Run the agent on each question.
    *   Submit all answers.
    *   Display the results and overall score.

The Gradio SDK version specified in the original Space config is `5.25.2`.

### B. Local Evaluation (using `gaia_scorer.py`)

You can run the agent on a local GAIA dataset file (e.g., downloaded in Step 4) and get scores.

1.  Ensure you have downloaded a GAIA dataset file (e.g., `gaia_level1_validation.jsonl`).
2.  Run the `gaia_scorer.py` script:

    ```bash
    python scripts/gaia_scorer.py \
        --data_path ./data/gaia/gaia_level1_validation.jsonl \
        --output_dir ./gaia_eval_output \
        --max_questions 10 # Optional: limit the number of questions
    ```

    *   `--data_path`: Path to your input GAIA dataset (JSONL format).
    *   `--output_dir`: Directory where results and failure analysis will be saved.
    *   `--max_questions` (optional): Process only the first N questions.
    *   `--results_filename` (optional): Filename for raw agent results (default: `agent_results_sequential.jsonl`).
    *   `--failures_filename` (optional): Filename for failure analysis (default: `failures_sequential.jsonl`).

3.  The script will:
    *   Process each question sequentially.
    *   Save raw agent outputs to `agent_results_sequential.jsonl` (or custom name) in the output directory.
    *   Score the predictions against the ground truth.
    *   Save details of incorrect answers to `failures_sequential.jsonl` (or custom name).
    *   Print a summary of total questions, correct answers, and accuracy.

## 🧩 Key Components Deep Dive

### Agents

*   **`ManagerAgent` (`gaia_agent/agents/manager_agent.py`):**
    *   The primary orchestrator. It's a `GAIACodeAgent`, meaning it plans and executes Python code snippets.
    *   It can use provided tools directly (e.g., `visualizer`, `inspect_file_as_text`).
    *   It can delegate tasks to `managed_agents`, such as the `WebResearchAgent`.
    *   Its behavior is heavily influenced by the prompts in `gaia_agent/prompts/gaia_code_agent.yaml`.
*   **`WebResearchAgent` (`gaia_agent/agents/web_research_agent.py`):**
    *   A specialized `GAIAToolCallingAgent` focused on web-based tasks.
    *   Comes pre-equipped with search tools, browser navigation tools, and the `TextInspectorTool` for downloaded files.
    *   Its behavior is guided by `gaia_agent/prompts/gaia_toolcalling_agent.yaml`.

### Core Tools

*   **Search Tools (`gaia_agent/tools/enhanced_search.py`):**
    *   `EnhancedGoogleSearchTool`, `EnhancedDuckDuckGoSearchTool`, `EnhancedWikipediaSearchTool`.
    *   These wrap `smolagents` tools, adding rate limiting and retry mechanisms for robustness.
*   **Text Web Browser (`gaia_agent/tools/text_web_browser.py`):**
    *   Provides `SimpleTextBrowser` (adapted from Autogen) and tools like:
        *   `VisitTool`: Visits a URL.
        *   `PageUpTool`/`PageDownTool`: Navigates page content.
        *   `FinderTool`/`FindNextTool`: Simulates Ctrl+F functionality.
        *   `ArchiveSearchTool`: Finds and visits pages on the Wayback Machine.
*   **File Conversion (`gaia_agent/tools/mdconvert.py`):**
    *   A powerful module (adapted from Autogen's `autogen-magentic-one`) that converts various file formats (PDF, DOCX, XLSX, PPTX, HTML, common audio/image types, ZIP archives, YouTube URLs) into Markdown text. This allows the LLM to "read" diverse content.
*   **Text Inspector (`gaia_agent/tools/text_inspector.py`):**
    *   `TextInspectorTool`: Uses `mdconvert` to read a local file and then employs an LLM to answer questions about its content or provide a summary. Handles various document types but not images directly (VQA tools are for images).
*   **Visual Question Answering (`gaia_agent/tools/visual_qa.py`):**
    *   `VisualQATool`: Uses `HuggingFaceM4/idefics2-8b-chatty` via `InferenceClient`.
    *   These tools allow the agent to answer questions about images.
*   **Speech-to-Text (`smolagents.default_tools.SpeechToTextTool`):**
    *   Integrated from `smolagents` to transcribe audio files.

### Prompts

*   Located in `gaia_agent/prompts/`.
*   `gaia_code_agent.yaml`: Defines the system prompt, planning instructions, and example interactions for the `ManagerAgent` (and other `GAIACodeAgent` instances). Emphasizes GAIA output formatting.
*   `gaia_toolcalling_agent.yaml`: Defines similar structures for `GAIAToolCallingAgent` instances like the `WebResearchAgent`.

### Configuration (`gaia_agent/config/settings.py`)

*   Uses `pydantic-settings` to manage application settings.
*   Loads from `.env` files and environment variables.
*   Defines API keys, model IDs, browser parameters, file paths, logging levels, and benchmark API URLs.
*   Includes validation, e.g., ensuring `GOOGLE_CLOUD_PROJECT` is set if Gemini models are used.

### Utilities

*   **`agent_setup.py`:** Contains the `setup_agent_system()` function, which is the main entry point for creating and configuring the `ManagerAgent` with its tools and sub-agents.
*   **`agent_wrapper.py`:** `GAIABenchmarkAgentWrapper` wraps the agent system, prepares the input task (augmenting questions, describing attached files using inspector/visualizer tools), and cleans the final agent output for GAIA compliance.
*   **`cookies.py`:** Contains a predefined list of cookies for various websites (YouTube, ResearchGate, GitHub, Archive.org, ORCID). This may help in accessing content from these sites that might otherwise be restricted or require interaction.

## 📝 TODO

*   **Extensive LLM Testing:** The system was primarily tested with Google Gemini models (specifically `gemini-2.0-flash` as per defaults). While the `LiteLLM` integration allows for configuration with other LLMs (OpenAI, Anthropic, etc.), these configurations have not been extensively tested. Performance and prompt compatibility may vary.
*   **Error Handling & Robustness:** Further improve error handling within tool executions and agent steps.
*   **Advanced Planning:** Explore more sophisticated planning mechanisms for the `ManagerAgent`.
*   **Memory Management:** Investigate more advanced memory management techniques for longer conversations or complex tasks.
*   **Tool Optimization:** Profile and optimize performance of individual tools, especially file conversion and VQA.
*   **Code Generation Security:** For `GAIACodeAgent`, enhance security around generated code execution if used in less controlled environments.



## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## 👤 Author

*   Marco Sassarini
```