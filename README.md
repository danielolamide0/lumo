# 🧸 Lumo AI - Enhanced AI Companion with Persistent Memory

Lumo is a sophisticated, memory-enabled AI companion designed specifically for children. Built with **Google Gemini 2.5**, **LangGraph**, **MongoDB Atlas**, **ChromaDB vector memory**, and **advanced timeline processing**, Lumo provides a safe, engaging, and educational chat experience that truly remembers every conversation and builds meaningful relationships over time.

## 🚀 **Project Overview**

### **What Makes Lumo Special**
- **🧠 Persistent Memory**: Remembers every conversation across sessions using LangGraph's official MongoDB checkpointer
- **🎭 Intelligent Routing**: AI-powered conversation mode detection (General, Game, Story, Learning)
- **📚 Timeline Memory**: Automatically creates narrative timelines of user relationships every 20 interactions
- **🔍 Vector Search**: Semantic search through conversation history using ChromaDB
- **⏰ Temporal Awareness**: Real-time timezone detection and date/time understanding
- **🔄 Dual-Write System**: Ensures data accessibility through both LangGraph checkpointer and browsable users collection
- **🎨 Modern UI**: Beautiful Streamlit interface with real-time chat and configuration panels

### **Live Demo Features**
- **Persistent Chat History**: Pick up conversations exactly where you left off
- **Memory Integration**: Lumo naturally references past conversations and personal details
- **Mode Detection**: Automatically switches between conversation, gaming, storytelling, and learning modes
- **Timeline Summaries**: Creates rich narrative summaries of your relationship over time
- **Real-time Processing**: Fast responses with background memory processing
- **Data Management**: Full user data control with export and deletion capabilities

## 🏗️ **System Architecture**

### **Core Technology Stack**
```
🧠 AI Engine:          Google Gemini 2.5 Flash Preview
🔄 Workflow:           LangGraph with MongoDB Checkpointer  
💾 Primary Storage:    MongoDB Atlas (Dual-Write System)
🔍 Vector Memory:      ChromaDB for semantic search
🎨 Interface:          Streamlit web application
⚡ Background Tasks:   Threading for timeline processing
🔐 Security:           Environment-based secrets management
```

### **Advanced Memory Architecture**

#### **1. LangGraph MongoDB Checkpointer (Primary)**
```python
# Official LangGraph MongoDB persistence
from langgraph.checkpoint.mongodb import MongoDBSaver

checkpointer = MongoDBSaver(
    client=MongoClient(MONGODB_URI),
    db_name="LUMO"
)

# Automatic state persistence with every interaction
ai_app = workflow.compile(checkpointer=checkpointer)
```

**Features:**
- ✅ **Automatic State Management**: Every conversation state automatically saved
- ✅ **Thread-based Sessions**: Each user gets isolated conversation threads  
- ✅ **Message History**: Complete conversation history with message ordering
- ✅ **Metadata Tracking**: Interaction counts, timestamps, current mode/emotion
- ✅ **Recovery Support**: Automatic recovery from interruptions

#### **2. Dual-Write Users Collection (Secondary)**
```python
# Synchronized user collection for easy browsing and analytics
def _sync_to_users_collection(self, state: LumoState):
    user_doc = {
        "_id": username,  # Username as primary key for consistency
        "username": username,
        "chats": formatted_chat_history,
        "timeline_summaries": timeline_data,
        "interaction_count": state["interaction_count"],
        "storage_notes": {
            "primary": "LangGraph MongoDB Checkpointer",
            "secondary": "Users Collection (Tracking & Timeline Access)"
        }
    }
    self.users_collection.replace_one({"_id": username}, user_doc, upsert=True)
```

**Benefits:**
- 📊 **Easy Browsing**: Direct MongoDB queries for analytics
- 🔄 **Data Backup**: Redundant storage for safety
- 📈 **Timeline Access**: Easy access to conversation summaries
- 🔗 **Migration Support**: Seamless migration from legacy formats
- 🆔 **Consistent IDs**: Username as document ID for reliable querying

#### **3. ChromaDB Vector Memory**
```python
# Timeline summaries stored as searchable vectors
class VectorMemoryManager:
    def store_timeline_memory(self, username: str, timeline: Dict[str, Any]):
        document = Document(
            page_content=timeline['story'],
            metadata={
                "username": username,
                "type": "timeline_memory", 
                "timestamp": timeline['updated_at'],
                "interactions": timeline['total_interactions']
            }
        )
        self.vector_store.add_documents([document])
```

**Capabilities:**
- 🔍 **Semantic Search**: Find relevant memories based on conversation context
- 📚 **Timeline Storage**: Only timeline summaries stored (not full conversations)
- ⚡ **Fast Retrieval**: Sub-second memory lookup during conversations
- 🎯 **Contextual Relevance**: Automatic memory enhancement based on current topic

### **Enhanced State Management**

#### **LumoState TypedDict Structure**
```python
class LumoState(TypedDict):
    # Core conversation - ONLY RECENT 20 MESSAGES (for efficiency)
    messages: List[Any]
    
    # User profile and metadata
    username: str
    user_profile: Dict[str, Any]
    user_timezone: Optional[str]
    
    # Memory tracking
    interaction_count: int
    timeline_memory: Dict[str, Any]  # ONLY current timeline summary
    vector_memory_metadata: Dict[str, Any]
    
    # Current context
    current_mode: str                # general|game|story|learning
    current_emotion: str             # happy|sad|excited|curious|etc.
    summary_context: Optional[str]   # Relevant memories for current conversation
    
    # System metadata
    created_at: str
    last_updated: str
```

## 🔄 **Conversation Workflow**

### **Step-by-Step Process**

#### **1. Message Processing**
```python
def process_message(self, username: str, message: str) -> Dict[str, Any]:
    # 1. Create thread configuration
    config = {"configurable": {"thread_id": f"enhanced_{username}"}}
    
    # 2. Retrieve existing state or create new
    existing_state = self._get_or_create_state(username, config)
    
    # 3. Add user message to conversation
    input_state = self._add_user_message(existing_state, message)
    
    # 4. Enhance with relevant memories
    enhanced_state = self._enhance_with_vector_memories(input_state, message)
    
    # 5. Process through LangGraph workflow
    response_state = self.ai_app.invoke(enhanced_state, config=config)
    
    # 6. Sync to users collection (dual-write)
    self._sync_to_users_collection(response_state)
    
    # 7. Background timeline processing (if needed)
    if response_state["interaction_count"] % 20 == 0:
        self._process_timeline_async(username, response_state)
    
    return response
```

#### **2. LangGraph Routing System**
```python
def _router(self, state: LumoState) -> str:
    # AI-powered analysis of user intent
    last_message = state["messages"][-1].content
    analysis = self._ai_analyze_intent_and_emotion(last_message)
    
    # Update state with detected mode and emotion
    state["current_mode"] = analysis["mode"]
    state["current_emotion"] = analysis["emotion"]
    
    # Route to appropriate specialized node
    return analysis["mode"]  # "general"|"game"|"story"|"learning"
```

#### **3. Memory Enhancement**
```python
def _enhance_state_with_memory(self, state: LumoState) -> LumoState:
    username = state["username"]
    current_message = state["messages"][-1].content
    
    # Retrieve relevant timeline memories
    relevant_memories = self.vector_memory.retrieve_relevant_memories(
        username, current_message, k=3
    )
    
    if relevant_memories:
        memory_context = "RELEVANT MEMORIES:\n" + "\n".join(relevant_memories)
        state["summary_context"] = memory_context
    
    return state
```

#### **4. Specialized Response Nodes**

**General Conversation Node:**
- Open-ended dialogue and emotional support
- Natural conversation with empathy and validation
- Emotional adaptation based on detected state

**Game Mode Node:**
- Interactive games (I Spy, 20 Questions, Word Association, Riddles)
- Game state tracking and progress management
- Age-appropriate challenges and encouragement

**Story Mode Node:**
- Collaborative storytelling and creative narratives
- Interactive plot development with user input
- Character development and world-building

**Learning Mode Node:**
- Educational content with child-friendly explanations
- Curiosity encouragement and follow-up questions
- Adaptive complexity based on understanding

#### **5. Timeline Processing (Background)**
```python
def _process_timeline_async(self, username: str, messages: List[Any], 
                          range_str: str, existing_timeline: Dict[str, Any]):
    # Runs in background thread for performance
    threading.Thread(
        target=self._update_timeline_summary,
        args=(username, messages, range_str, existing_timeline),
        daemon=True
    ).start()

def _update_timeline_summary(self, username, messages, range_str, timeline):
    # 1. Create conversation summary from message range
    temp_summary = self._create_conversation_summary(username, messages, range_str)
    
    # 2. Update timeline narrative with time-aware integration
    updated_timeline = self._integrate_summary_into_timeline(temp_summary, timeline)
    
    # 3. Store updated timeline in ChromaDB for semantic search
    self.vector_memory.store_timeline_memory(username, updated_timeline)
    
    # 4. Update users collection with new timeline
    self._update_user_timeline(username, updated_timeline)
```

### **Performance Optimizations**

#### **1. Fast Response Times**
- **Immediate Response**: AI response returned before timeline processing
- **Background Processing**: Timeline updates happen asynchronously
- **Message Trimming**: Only recent 20 messages kept in active state
- **Vector Caching**: Frequently accessed memories cached for speed

#### **2. Memory Efficiency**
- **Selective Storage**: Only timeline summaries in vector database
- **Automatic Cleanup**: Temporary summaries deleted after timeline integration
- **State Optimization**: Minimal data in LangGraph state for fast serialization

#### **3. Scalability Features**
- **Thread Isolation**: Each user has independent conversation threads
- **Async Processing**: Non-blocking timeline and memory operations
- **Database Indexing**: Optimized MongoDB indexes for fast queries
- **Connection Pooling**: Efficient database connection management

## 🎯 **AI-Powered Features**

### **1. Intent Detection & Emotion Analysis**
```python
def _ai_analyze_intent_and_emotion(self, user_message: str) -> dict:
    analysis_prompt = f"""
    Analyze this child's message and classify:
    
    User message: "{user_message}"
    
    MODES:
    - "general": Casual conversation, personal sharing, emotional support
    - "game": Wants to play games, interactive activities, fun challenges  
    - "story": Wants stories, narratives, creative tales, imaginative content
    - "learning": Educational questions, how/why/what, wants to learn something
    
    EMOTIONS:
    - "happy": Joyful, positive, excited, cheerful
    - "sad": Upset, disappointed, down, hurt feelings
    - "curious": Wondering, asking questions, eager to explore
    - "frustrated": Struggling, annoyed, having difficulty
    - "neutral": Calm, normal, no strong emotion
    
    Return JSON: {{"mode": "X", "emotion": "Y", "confidence": 0.9, "reasoning": "..."}}
    """
    
    # Uses Gemini AI for classification with fallback to keyword matching
```

### **2. Memory-Enhanced Responses**
```python
def _call_llm_with_enhanced_context(self, state: LumoState, interaction_type: str):
    # Build comprehensive prompt with:
    # - Core personality and safety rules
    # - Conversation foundation skills  
    # - Mode-specific specialization
    # - Current emotional context
    # - Relevant timeline memories
    # - Temporal awareness (current date/time)
    
    enhanced_prompt = f"""
    {self.core_identity}
    
    {self.chat_foundation}
    
    {self.mode_prompts[interaction_type]}
    
    EMOTION: {state['current_emotion']}
    
    {state.get('summary_context', '')}
    
    CURRENT TIME: {datetime.now().strftime('%A, %B %d, %Y at %I:%M %p')}
    """
```

### **3. Temporal Intelligence**
```python
def _add_temporal_context_to_prompt(self, base_prompt: str, user_timezone: str = "UTC"):
    # Adds real-time awareness to every conversation
    current_time = datetime.now(pytz.timezone(user_timezone))
    
    temporal_context = f"""
    CURRENT TEMPORAL AWARENESS:
    - Today's Date: {current_time.strftime('%A, %B %d, %Y')}
    - Current Time: {current_time.strftime('%I:%M %p')}
    - Timezone: {user_timezone}
    
    IMPORTANT: When users mention "today", "yesterday", "now", etc., 
    use these ACTUAL dates and times in your responses.
    """
    
    return f"{base_prompt}\n\n{temporal_context}"
```

## 🔧 **Setup & Installation**

### **1. Prerequisites**
```bash
# Python 3.9+ required
python --version

# Virtual environment recommended
python -m venv lumo_env
source lumo_env/bin/activate  # On Windows: lumo_env\Scripts\activate
```

### **2. Install Dependencies**
```bash
# Clone the repository
git clone <repository-url>
cd lumo-1

# Install required packages
pip install -r requirements.txt
```

### **3. Environment Configuration**

#### **Create `.env` file:**
```bash
# Google AI Configuration
GOOGLE_API_KEY=your_gemini_api_key_here
MODEL_NAME=gemini-2.5-flash-preview-04-17

# MongoDB Configuration  
MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/
DATABASE_NAME=LUMO

# Optional: Timezone settings
DEFAULT_TIMEZONE=UTC
```

#### **Create `.streamlit/secrets.toml`:**
```toml
# Streamlit Secrets (for deployment)
GEMINI_API_KEY = "your_gemini_api_key_here"
MODEL_NAME = "gemini-2.5-flash-preview-04-17"
MONGODB_URI = "mongodb+srv://username:password@cluster.mongodb.net/"
DATABASE_NAME = "LUMO"
```

### **4. MongoDB Setup**

#### **Create MongoDB Atlas Account:**
1. Go to [MongoDB Atlas](https://www.mongodb.com/cloud/atlas)
2. Create free cluster
3. Create database user with read/write permissions
4. Add your IP address to whitelist (or use 0.0.0.0/0 for development)
5. Get connection string and add to environment variables

#### **Required Collections:**
The application will automatically create these collections:
- `checkpoints`: LangGraph state storage (automatic)
- `checkpoint_writes`: LangGraph write operations (automatic)  
- `users`: User data and timeline summaries (dual-write)

### **5. Google AI Setup**
1. Go to [Google AI Studio](https://aistudio.google.com)
2. Create API key for Gemini
3. Add API key to environment variables
4. Ensure Gemini API is enabled for your project

### **6. Run the Application**
```bash
# Start Streamlit application
streamlit run streamlit_app.py --server.port 8507

# Access at: http://localhost:8507
```

## 🎮 **Usage Guide**

### **1. First Time Setup**
1. Open `http://localhost:8507` in your browser
2. Enter a username (no registration required)
3. Start chatting with Lumo immediately
4. Your conversation history will automatically persist

### **2. Core Features**

#### **Persistent Conversations**
- All conversations automatically saved across sessions
- Pick up exactly where you left off
- Complete message history maintained
- Timeline summaries created every 20 interactions

#### **Intelligent Mode Detection** 
- **General**: "How was your day?" → Casual conversation and emotional support
- **Game**: "Let's play a game!" → Interactive games and fun activities
- **Story**: "Tell me a story" → Creative storytelling and narratives  
- **Learning**: "How do airplanes fly?" → Educational explanations

#### **Memory Integration**
- Lumo naturally references past conversations
- Personal details remembered and mentioned contextually
- Timeline memories surface during relevant discussions
- Emotional context maintained across sessions

#### **Configuration Panel**
- Access via "⚙️ Configure Lumo's Architecture" expander
- Modify core personality, chat foundation, and mode-specific behaviors
- Real-time prompt combination preview
- Restart Lumo to apply changes

### **3. Advanced Features**

#### **Timeline Memory Exploration**
Every 20 interactions, Lumo automatically:
1. Creates detailed conversation summary
2. Updates relationship timeline narrative
3. Stores timeline in searchable vector memory
4. References timeline in future conversations

#### **Data Management**
- View user profile and interaction statistics
- Delete all user data with confirmation
- Export conversation history (through direct MongoDB access)
- Monitor storage across both collections

#### **Temporal Intelligence**
- Lumo knows the current date and time
- Handles timezone-aware conversations
- Processes relative time references ("yesterday", "today", "now")
- Maintains conversation continuity across time gaps

## 🗂️ **Database Schema**

### **LangGraph Collections (Automatic)**

#### **`checkpoints` Collection:**
```javascript
{
  "_id": ObjectId("..."),
  "thread_id": "enhanced_username",
  "checkpoint_id": "1ef234ab-5678-1234-ef56-789012345678", 
  "parent_checkpoint_id": null,
  "values": {
    "messages": [...],           // Recent 20 messages only
    "username": "username",
    "interaction_count": 25,
    "timeline_memory": {...},    // Current timeline summary
    "current_mode": "general",
    "current_emotion": "happy",
    "created_at": "2024-01-15T10:30:00Z",
    "last_updated": "2024-01-15T11:45:30Z"
  },
  "metadata": {...}
}
```

#### **`checkpoint_writes` Collection:**
```javascript
{
  "_id": ObjectId("..."),
  "thread_id": "enhanced_username", 
  "checkpoint_id": "1ef234ab-5678-1234-ef56-789012345678",
  "task_id": "enhance_and_route",
  "idx": 0,
  "channel": "messages",
  "value": [...],               // Message content
  "type": "update"
}
```

### **Users Collection (Dual-Write)**
```javascript
{
  "_id": "username",            // Username as primary key
  "username": "username",
  "chats": [
    {
      "user_input": "Hello Lumo!",
      "ai_response": "Hi there! How are you feeling today?",
      "timestamp": "2024-01-15T10:30:15Z",
      "ai_timestamp": "2024-01-15T10:30:18Z",
      "interaction_id": 1
    }
  ],
  "profile": {
    "preferences": {},
    "interests": [],
    "age_appropriate_content": true
  },
  "interaction_count": 25,
  "timeline_summaries": {
    "story": "Comprehensive narrative of relationship progression...",
    "created_at": "2024-01-15T10:30:00Z", 
    "updated_at": "2024-01-15T11:45:30Z",
    "total_interactions": 25,
    "summaries_processed": 1,
    "last_interaction_time": "2024-01-15T11:45:30Z"
  },
  "summaries": [],              // Legacy field for migration compatibility
  "current_mode": "general",
  "current_emotion": "happy", 
  "created_at": "2024-01-15T10:30:00Z",
  "updated_at": "2024-01-15T11:45:30Z",
  "email": "username@lumo.ai",  // Placeholder
  "user_timezone": "UTC",
  "vector_memory_metadata": {
    "timeline_memories_count": 1
  },
  "storage_notes": {
    "primary": "LangGraph MongoDB Checkpointer",
    "secondary": "Users Collection (Tracking & Timeline Access)",
    "vector_memory": "ChromaDB Timeline Summaries Only",
    "format": "Compatible with original users collection schema"
  }
}
```

### **ChromaDB Vector Storage**
```python
# Documents stored as vectors with metadata
{
  "id": "username_timeline_1",
  "content": "Rich narrative timeline of conversations and relationship...",
  "metadata": {
    "username": "username",
    "type": "timeline_memory",
    "timestamp": "2024-01-15T11:45:30Z",
    "interactions": 25
  },
  "embedding": [0.123, 0.456, 0.789, ...]  # 384-dimensional vector
}
```

## 🔒 **Security & Privacy**

### **Data Protection**
- ✅ **Environment-based Secrets**: All API keys and database URLs in environment variables
- ✅ **Git Ignore Protection**: Sensitive files automatically excluded from version control
- ✅ **User Isolation**: Each user's data completely separated by username
- ✅ **No Personal Data Required**: Only username needed to start chatting
- ✅ **Data Deletion**: Complete user data deletion available through interface

### **Child Safety Features**
- ✅ **Content Filtering**: AI trained on child-appropriate responses
- ✅ **Emotional Support**: Recognizes and responds to emotional distress
- ✅ **Educational Focus**: Encourages learning and curiosity
- ✅ **Positive Reinforcement**: Always encouraging and supportive
- ✅ **Safe Gaming**: Age-appropriate games and challenges

### **Technical Security**
- ✅ **Secure Connections**: MongoDB Atlas with SSL/TLS encryption
- ✅ **API Security**: Google AI API with proper authentication
- ✅ **Session Management**: Streamlit secure session handling
- ✅ **Input Validation**: Proper sanitization of user inputs
- ✅ **Error Handling**: Graceful error handling without data exposure

## 🚀 **Performance Metrics**

### **Response Times**
- **Average Response**: 2-3 seconds for simple conversations
- **Complex Responses**: 4-6 seconds for educational/story content
- **Memory Retrieval**: <1 second for relevant memory lookup
- **Timeline Processing**: Background (non-blocking) processing
- **Database Queries**: <500ms for user data retrieval

### **Memory Efficiency**
- **Active State Size**: ~20 recent messages per user
- **Timeline Storage**: Compressed narrative summaries only
- **Vector Embeddings**: 384-dimensional for semantic search
- **Database Indexing**: Optimized indexes on username and timestamp fields

### **Scalability**
- **Concurrent Users**: Designed for 100+ simultaneous conversations
- **Database Performance**: MongoDB Atlas handles thousands of users
- **Vector Search**: ChromaDB efficient for millions of embeddings
- **Background Processing**: Non-blocking timeline updates

## 🛠️ **Development & Customization**

### **Adding New Conversation Modes**
```python
# 1. Add to MODE_SPECIFIC_PROMPTS in ai_toy_agent.py
MODE_SPECIFIC_PROMPTS["creative"] = """
CURRENT MODE: Creative Expression
FOCUS: Art, music, creative projects and imagination
SPECIALIZED BEHAVIOR:
- Encourage artistic and musical expression
- Provide creative prompts and ideas
- Support imaginative projects
"""

# 2. Update LangGraph workflow
def _setup_enhanced_graph(self):
    self.workflow.add_node("creative", 
        lambda state: self._call_llm_with_enhanced_context(state, "creative"))
    
# 3. Update router to detect new mode
def _ai_analyze_intent_and_emotion(self, user_message: str):
    # Add "creative" to MODES in analysis prompt
```

### **Customizing Memory Behavior**
```python
# Modify timeline processing frequency (currently every 20 interactions)
TIMELINE_PROCESSING_FREQUENCY = 30  # Process every 30 interactions

# Customize memory retrieval count
MEMORY_RETRIEVAL_COUNT = 5  # Retrieve 5 relevant memories instead of 3

# Adjust memory relevance threshold
MEMORY_RELEVANCE_THRESHOLD = 0.4  # Higher threshold for more selective memory
```

### **Adding New AI Capabilities**
```python
# Example: Add image description capability
def _analyze_image(self, image_data):
    # Integrate with Google's vision API
    # Add to conversation context
    # Enable multimodal conversations
```

## 📊 **Monitoring & Analytics**

### **Built-in Analytics**
- **User Interaction Counts**: Track engagement per user
- **Mode Distribution**: See which conversation modes are most popular  
- **Timeline Generation**: Monitor memory system performance
- **Error Tracking**: Built-in error logging and recovery

### **Database Monitoring**
```python
# Get user overview
agent = EnhancedLumoAgent()
users_overview = agent.get_all_users_overview()

# Monitor specific user
user_info = agent.get_user_info("username")
print(f"Interactions: {user_info['interaction_count']}")
print(f"Timeline memories: {user_info['vector_memory_count']}")
```

### **Performance Monitoring**
```python
# Check system status
def system_health_check():
    # Test MongoDB connection
    # Verify ChromaDB accessibility  
    # Check Google AI API status
    # Validate LangGraph workflow
```

## 🤝 **Contributing**

### **Development Setup**
```bash
# Clone and setup development environment
git clone <repository-url>
cd lumo-1
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Create development configuration
cp .env.example .env
# Add your API keys and database URLs

# Run development server
streamlit run streamlit_app.py --server.port 8507
```

### **Code Structure Guidelines**
- **ai_toy_agent.py**: Core AI logic and database interactions
- **streamlit_app.py**: User interface and session management
- **requirements.txt**: Dependency management
- **README.md**: Documentation and setup instructions

### **Testing Procedures**
```bash
# Test new user creation
python -c "
from ai_toy_agent import EnhancedLumoAgent
agent = EnhancedLumoAgent()
result = agent.process_message('test_user', 'Hello Lumo!')
print('✅ New user test:', result['success'])
"

# Test memory retrieval  
python -c "
from ai_toy_agent import EnhancedLumoAgent
agent = EnhancedLumoAgent()
info = agent.get_user_info('existing_user')
print('✅ Memory test:', info.get('vector_memory_count', 0))
"
```

## 📚 **Additional Resources**

### **Documentation Links**
- [LangGraph Documentation](https://python.langchain.com/docs/langgraph)
- [MongoDB Atlas Setup Guide](https://docs.atlas.mongodb.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Google AI Studio](https://aistudio.google.com/)

### **Troubleshooting**
- **Port 8507 in use**: Kill existing process with `pkill -f streamlit`
- **MongoDB connection issues**: Check IP whitelist and connection string
- **API key errors**: Verify environment variables and .streamlit/secrets.toml
- **Memory errors**: Restart ChromaDB by deleting `chroma_lumo_memory` folder

### **Support**
For technical support or questions about the Lumo AI system:
1. Check the troubleshooting section above
2. Review the comprehensive logging in the application console
3. Verify all environment variables are properly configured
4. Test individual components (MongoDB, Gemini AI, ChromaDB) separately

---

**Built with ❤️ for creating meaningful AI relationships that remember and grow over time.** 