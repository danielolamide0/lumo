# 🧸 Lumo AI - Advanced AI Companion for Children

Lumo is a sophisticated, memory-enabled AI companion designed specifically for children. Built with Google's Gemini AI, LangGraph workflow orchestration, and persistent MongoDB storage, Lumo provides a safe, engaging, and educational chat experience that remembers every conversation.

## ✨ Key Features

### 🧠 **Advanced AI Architecture**
- **Core Identity + Chat Foundation + Specialized Modes** - Modular prompt architecture
- **Dynamic Intent Detection** - AI-powered routing to Game, Story, Learning, or General modes
- **Emotional Intelligence** - Adapts responses based on detected emotional state
- **Natural Memory Usage** - References past conversations naturally without explicit "I remember" statements

### 💾 **Persistent Memory System**
- **MongoDB Atlas Integration** - All conversations stored in cloud database
- **User-Centric Storage** - Each user gets their own profile and conversation history
- **Cross-Session Continuity** - Pick up conversations exactly where you left off
- **Real-time Chat Loading** - Previous conversations appear instantly on login

### 👤 **User Management**
- **Username-Based Authentication** - Simple login system for children
- **Individual User Profiles** - Separate conversation histories for each user
- **User Dashboard** - Shows total chats, storage status, and member since date
- **Session Management** - Login/logout functionality with state persistence

### 🎭 **Specialized Interaction Modes**
- **General Chat** - Open-ended dialogue and emotional support
- **Game Mode** - Interactive games (I Spy, 20 Questions, Word Association, etc.)
- **Story Mode** - Collaborative storytelling and creative narratives
- **Learning Mode** - Educational exploration with child-friendly explanations

### 🛡️ **Safety & Child-Friendly Design**
- Age-appropriate language and content filtering
- Emotional validation and supportive responses
- Comprehensive error handling with child-friendly messages
- Safe conversation boundaries and guidelines

## 🏗️ **Technical Architecture**

### **Core Components**
```
/lumo/
├── .streamlit/
│   └── secrets.toml          # Streamlit configuration (NOT in git)
├── ai_toy_agent.py          # Main AI agent with MongoDB integration
├── streamlit_app.py         # User interface and authentication
├── requirements.txt         # Python dependencies
├── .gitignore              # Protects sensitive files
└── README.md               # This documentation
```

### **Technology Stack**
- **AI Model**: Google Gemini 2.5 Flash Preview
- **Framework**: LangGraph for conversation workflows
- **Database**: MongoDB Atlas for persistent storage
- **Interface**: Streamlit for web-based chat
- **Memory**: User-centric conversation persistence
- **Routing**: AI-powered intent and emotion detection

### **Database Structure**
```javascript
// MongoDB Atlas - LUMO Database - users Collection
{
  "_id": "username",
  "username": "username",
  "email": "username@example.com",
  "created_at": "2024-01-15T...",
  "chats": [
    {
      "user_input": "Hello!",
      "ai_response": "Hi there! How are you?",
      "timestamp": "2024-01-15T..."
    }
  ],
  "profile": {},
  "summaries": []
}
```

## 🚀 **Setup and Installation**

### **Prerequisites**
- Python 3.8+
- Google Gemini API Key
- MongoDB Atlas Account

### **1. Clone Repository**
```bash
git clone https://github.com/danielolamide0/lumo.git
cd lumo
git checkout memorymongodb
```

### **2. Environment Setup**
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### **3. Configuration**
Create `.streamlit/secrets.toml`:
```toml
MONGODB_URI = "your_mongodb_atlas_connection_string"
GEMINI_API_KEY = "your_google_gemini_api_key"
MODEL_NAME = "gemini-2.5-flash-preview-04-17"
```

### **4. MongoDB Atlas Setup**
1. Create MongoDB Atlas account
2. Create cluster and database named `LUMO`
3. Create collection named `users`
4. Get connection string and add to secrets.toml

## 🎮 **Running Lumo**

### **Web Interface (Recommended)**
```bash
streamlit run streamlit_app.py
```
Navigate to `http://localhost:8501`

### **Command Line Interface**
```bash
python ai_toy_agent.py
```

## 🎯 **Usage**

1. **Login**: Enter your username to create/access your profile
2. **Chat**: Start conversing with Lumo naturally
3. **Modes**: Lumo automatically detects when you want to:
   - Play games ("Let's play something!")
   - Hear stories ("Tell me a story!")
   - Learn something ("How do rockets work?")
   - Just chat ("How was your day?")
4. **Memory**: Lumo remembers everything across sessions
5. **Logout**: Use the logout button to switch users

## ⚙️ **Customization**

### **Prompt Engineering**
Use the built-in configuration interface to customize:
- **Core Identity**: Lumo's fundamental personality
- **Chat Foundation**: Shared conversational abilities
- **Mode Specializations**: Behavior for each interaction mode

### **Architecture Preview**
View how Core Identity + Chat + Specialized Mode prompts combine for different interaction types.

## 🔧 **Advanced Features**

### **Memory System**
- **Natural Referencing**: Uses conversation history seamlessly
- **Cross-Session Persistence**: MongoDB Atlas storage
- **User-Specific**: Separate memories for each user
- **Real-time Updates**: Conversations saved instantly

### **AI Routing**
- **Intent Detection**: Game/Story/Learning/General classification
- **Emotion Recognition**: Happy/Sad/Excited/Curious/etc.
- **Dynamic Adaptation**: Emotional context influences responses
- **Fallback Systems**: Multiple layers of error handling

### **Error Handling**
- **Child-Friendly Messages**: Age-appropriate error responses
- **Connection Resilience**: Handles network/database issues
- **API Quotas**: Graceful handling of rate limits
- **Authentication**: Secure API key management

## 📊 **Monitoring & Debugging**

The system provides comprehensive logging:
- User creation and authentication
- Chat message storage and retrieval
- AI routing decisions and confidence levels
- MongoDB connection status
- Error tracking and fallback usage

## 🛡️ **Security**

- **API Key Protection**: Stored in secrets.toml (not in git)
- **MongoDB Credentials**: Secured in configuration
- **Child Safety**: Content filtering and appropriate responses
- **Session Management**: Secure user state handling

## 🚀 **Deployment**

For production deployment:
1. Set up Streamlit Cloud or similar service
2. Configure secrets in deployment environment
3. Ensure MongoDB Atlas is accessible
4. Set up monitoring and alerts

## 🤝 **Contributing**

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📝 **License**

This project is open source and available under the MIT License.

## 🙏 **Acknowledgments**

- Google Gemini AI for the powerful language model
- LangGraph for conversation workflow orchestration
- MongoDB Atlas for reliable cloud database
- Streamlit for the intuitive web interface

---

**Created by [@danielolamide0](https://github.com/danielolamide0)**  
**Last Updated**: January 2024  
**Version**: MongoDB Memory Integration
