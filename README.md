# Lumo AI Toy - Interactive Chat Interface

This project implements "Lumo," a friendly and playful AI companion designed to interact with children. Lumo uses Google's Gemini Pro model to chat, play games, tell stories, and learn new things with its young friends.

## Features

- **Conversational AI:** Lumo can engage in natural, child-friendly conversations.
- **Playful Personality:** Defined by a system prompt that encourages cheerful, patient, and curious interactions.
- **Contextual Memory:** Remembers previous parts of the conversation (using in-memory LangGraph checkpointer).
- **Extensible:** Built with LangGraph, allowing for future expansion with more complex behaviors or tools.

## Project Structure

```
/lumo/
├── .env                 # For environment variables (GOOGLE_API_KEY)
├── .venv/               # Virtual environment (recommended)
├── ai_toy_agent.py      # Main script for the AI toy agent
├── streamlit_app.py     # Streamlit interface
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## Setup and Installation

### 1. Prerequisites

- Python 3.8 or higher
- Google Gemini API Key

### 2. Clone the Repository

```bash
git clone https://github.com/danielolamide0/lumo.git
cd lumo
```

### 3. Set Up Environment Variables

Create a `.env` file in the root directory with your Gemini API Key:

```env
GOOGLE_API_KEY=your-real-api-key-here
```

### 4. Create a Virtual Environment (Recommended)

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

## Running Lumo

Start the Streamlit interface:

```bash
streamlit run streamlit_app.py
```

Open your web browser and navigate to the URL shown in the terminal (typically http://localhost:8501)

### Alternative: Command Line Interface

You can also run Lumo directly from the command line:

```bash
python ai_toy_agent.py
```

## Customizing Lumo

You can customize Lumo's personality by modifying the system prompt in the web interface. Click on "⚙️ Configure Lumo's System Prompt" to make changes.

## License

This project is open source and available under the MIT License.

---
Created by [@danielolamide0](https://github.com/danielolamide0)
Last Updated: 2025-05-31