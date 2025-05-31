# AI Toy Agent - Lumo

This project implements "Lumo," a friendly and playful AI companion designed to interact with children. Lumo uses Google's Gemini 1.5 Flash model via Vertex AI to chat, play games, tell stories, and learn new things with its young friends.

## Features

- **Conversational AI:** Lumo can engage in natural, child-friendly conversations.
- **Playful Personality:** Defined by a system prompt that encourages cheerful, patient, and curious interactions.
- **Contextual Memory:** Remembers previous parts of the conversation (using in-memory LangGraph checkpointer).
- **Extensible:** Built with LangGraph, allowing for future expansion with more complex behaviors or tools.

## Project Structure

```
/mnt/d/lumo/lumo/
├── .env                 # For environment variables (GOOGLE_APPLICATION_CREDENTIALS, GCLOUD_PROJECT, GCLOUD_LOCATION)
├── .venv/               # Virtual environment (recommended)
├── ai_toy_agent.py      # Main script for the AI toy agent
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

## Setup and Installation

### 1. Prerequisites

- Python 3.8 or higher
- Access to Google Cloud Platform (GCP) and a configured project.
- Vertex AI API enabled in your GCP project.
- A Google Cloud service account key JSON file with appropriate permissions for Vertex AI.

### 2. Clone the Repository (if applicable)

If this project were in a Git repository, you would clone it:
```bash
git clone <repository_url>
cd <repository_directory>
```

### 3. Set Up Environment Variables

Create a `.env` file in the root directory (`/mnt/d/lumo/lumo/`) with your Google Cloud credentials:

```env
GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/service-account-file.json"
GCLOUD_PROJECT="your-gcp-project-id"
GCLOUD_LOCATION="your-gcp-region" # e.g., us-central1
```

Replace the placeholder values with your actual credentials and project information.

Alternatively, you can set these environment variables directly in your terminal:
```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/service-account-file.json"
export GCLOUD_PROJECT="your-gcp-project-id"
export GCLOUD_LOCATION="your-gcp-region"
```

### 4. Create a Virtual Environment (Recommended)

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 5. Install Dependencies

Install the required Python packages using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```
The `requirements.txt` should contain:
```
langchain-core
langchain-google-vertexai
langsmith
langgraph
python-dotenv
```

## Running the AI Toy Agent

Once the setup is complete, you can run the AI toy agent script:

```bash
python ai_toy_agent.py
```