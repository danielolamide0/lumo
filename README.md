# Lumo AI Toy - Enhanced Interactive AI Companion

This project implements "Lumo," an advanced AI companion designed specifically for children. Lumo uses Google's Gemini AI with sophisticated routing, emotional intelligence, and adaptive responses to create engaging, safe, and educational interactions.

## 🌟 Key Features

### **AI-Powered Intelligence**
- **Dynamic Intent Detection:** Uses Gemini AI to intelligently analyze user messages and determine interaction type
- **Emotional Intelligence:** Detects emotions (happy, sad, excited, curious, confused, tired, frustrated, neutral) and adapts responses accordingly
- **Smart Routing:** No keyword matching - pure AI analysis for natural conversation flow

### **Multi-Mode Interactions**
- **Chat Mode:** General conversation and companionship with emotional awareness
- **Game Mode:** Interactive games (I Spy, 20 Questions, Word Association, Riddles) adapted to emotional state
- **Story Mode:** Dynamic storytelling with user choices and emotional adaptation
- **Learning Mode:** Educational explanations tailored to curiosity level and emotional state

### **Best Practice Architecture**
- **Core Identity:** Shared personality, safety rules, and communication style across all modes
- **Mode-Specific Prompts:** Specialized instructions for each interaction type
- **Emotional Adaptation:** Each mode adapts behavior based on detected emotions
- **Layered System:** Core + Mode + Emotion = Complete AI Response

### **Advanced Technical Features**
- **LangGraph Workflow:** Parallel processing with conditional routing for optimal performance
- **Comprehensive Fallback System:** Three-layer fallback (AI Analysis → Text Parsing → Keyword Matching)
- **Analysis Caching:** Optimized performance with intelligent caching to avoid duplicate AI calls
- **Error Handling:** Robust error recovery ensures system always responds gracefully

### **Enhanced User Interface**
- **Multi-Tab Configuration:** Separate editors for core identity and each mode
- **Real-Time Preview:** View combined prompts before applying changes
- **Prompt Statistics:** Monitor character counts and system performance
- **Interactive Testing:** Built-in examples and testing guidance

## 🏗️ Project Architecture

```
User Input
    ↓
┌─────────────────────────────────────┐
│           LangGraph                 │
│  ┌─────────────────────────────────┐│
│  │ 1. AI Analysis (Intent+Emotion) ││  ← Gemini AI
│  │ 2. Conditional Router           ││  ← LangGraph
│  │ 3. Mode-Specific Node           ││  ← Our Logic
│  │ 4. Emotional Adaptation         ││  ← Our Logic
│  │ 5. Combined Response            ││  ← Gemini AI
│  │ 6. Memory Persistence           ││  ← LangGraph
│  └─────────────────────────────────┘│
└─────────────────────────────────────┘
    ↓
Emotionally Aware Response
```

### **Prompt Architecture**
```
Final Prompt = Core Identity + Mode Instructions + Emotional Context
```

Example:
- **Core Identity:** "You are Lumo, friendly and safe..."
- **Game Mode:** "You're a game master who suggests I Spy..."
- **Emotional Context:** "The child seems excited, match their energy..."
- **Result:** Enthusiastic game suggestions perfect for an excited child

## 📁 Project Structure

```
/lumo/
├── .env                 # Environment variables (GOOGLE_API_KEY, MODEL_NAME)
├── .venv/               # Virtual environment (recommended)
├── ai_toy_agent.py      # Enhanced AI agent with emotional intelligence
├── streamlit_app.py     # Multi-tab configuration interface
├── requirements.txt     # Python dependencies
├── .gitignore           # Git ignore file (excludes .env for security)
└── README.md           # This file
```

## 🚀 Setup and Installation

### 1. Prerequisites

- Python 3.8 or higher
- Google Gemini API Key ([Get one here](https://makersuite.google.com/app/apikey))

### 2. Clone the Repository

```bash
git clone https://github.com/danielolamide0/lumo.git
cd lumo
```

### 3. Set Up Environment Variables

Create a `.env` file in the root directory:

```env
GOOGLE_API_KEY=your-gemini-api-key-here
MODEL_NAME=gemini-1.5-flash
```

### 4. Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

## 🎮 Running Lumo

### Web Interface (Recommended)

```bash
streamlit run streamlit_app.py
```

Open your browser to the displayed URL (typically http://localhost:8501)

### Command Line Interface

```bash
python ai_toy_agent.py
```

## 🔧 Configuration & Customization

### **Multi-Tab Interface**
1. **Core Identity Tab:** Edit Lumo's fundamental personality and safety rules
2. **Chat Mode Tab:** Customize general conversation behavior
3. **Game Mode Tab:** Configure game-specific instructions and available games
4. **Story Mode Tab:** Set up storytelling behavior and story types
5. **Learning Mode Tab:** Define educational approach and teaching style
6. **View Combined Tab:** Preview how prompts combine for each mode

### **Testing Different Modes**
- **"I'm bored, let's do something fun!"** → Game Mode + Neutral/Bored emotion
- **"I had a bad day, tell me a happy story"** → Story Mode + Sad emotion
- **"I'm so excited! How do rockets work?!"** → Learning Mode + Excited emotion
- **"Hi! I'm feeling great today!"** → Chat Mode + Happy emotion

## 🧠 AI Analysis System

### **Intent Detection**
The system analyzes user messages to determine:
- **Chat:** General conversation, sharing feelings, asking about Lumo
- **Game:** Wanting to play, do activities, have fun interactions
- **Story:** Requesting narratives, roleplay, creative stories
- **Learning:** Asking how things work, educational curiosity

### **Emotion Detection**
Lumo recognizes and adapts to:
- **Happy/Excited:** Matches energy with enthusiasm
- **Sad/Frustrated:** Provides extra comfort and support
- **Curious:** Encourages questions and exploration
- **Tired:** Uses calmer, gentler responses
- **Confused:** Breaks things down simply
- **Neutral:** Standard friendly interaction

### **Fallback System**
1. **Primary:** AI analysis using Gemini for intent and emotion
2. **Secondary:** Text parsing if AI doesn't return JSON
3. **Tertiary:** Keyword matching if API fails
4. **Always responds** gracefully regardless of analysis success

## 🛡️ Safety & Security

- **Child-Safe Design:** All responses filtered for age-appropriate content
- **No Personal Information:** Never requests personal details
- **Positive Reinforcement:** Always encouraging and supportive
- **Emotional Support:** Recognizes distress and guides to adult help
- **Secure API Keys:** Environment variables prevent accidental exposure

## 🔄 Technical Features

### **LangGraph Integration**
- **Conditional Routing:** Intelligent flow control based on AI analysis
- **State Management:** Conversation memory with thread-based persistence
- **Parallel Processing:** Efficient workflow execution
- **Error Recovery:** Graceful handling of API failures

### **Performance Optimization**
- **Analysis Caching:** Avoids duplicate AI calls for same message
- **Efficient Prompting:** Combines prompts optimally to reduce token usage
- **Fallback Layers:** Ensures fast response even if primary analysis fails

## 🚀 Deployment Options

### **Local Development**
- Use `.env` file for API keys
- Run with `streamlit run streamlit_app.py`

### **Streamlit Cloud**
- Set API keys in Streamlit Secrets
- Automatic deployment from GitHub repository
- Environment variables handled securely

### **Custom Deployment**
- Compatible with any Python hosting platform
- Requires Streamlit and LangChain dependencies
- API key configuration via environment variables

## 🤝 Contributing

This project is designed for educational purposes and child safety. Contributions should maintain:
- Child-appropriate content standards
- Robust error handling
- Clear documentation
- Safety-first design principles

## 📄 License

This project is open source and available under the MIT License.

## 🎯 Roadmap

- [ ] Add more game types with state persistence
- [ ] Implement story branching with memory
- [ ] Add voice interaction capabilities
- [ ] Create parent dashboard for monitoring
- [ ] Add multilingual support
- [ ] Implement learning progress tracking

---

**Created by [@danielolamide0](https://github.com/danielolamide0)**  
**Enhanced with AI-Powered Intelligence & Emotional Awareness**  
**Last Updated: 2025-06-01**