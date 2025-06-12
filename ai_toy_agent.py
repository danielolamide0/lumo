import os
import uuid
import streamlit as st
from dotenv import load_dotenv
from typing import Dict, List, Any, Tuple, Annotated, Optional, TypedDict
from enum import Enum
from datetime import datetime, timedelta
import pytz
import json
import pickle
import base64

# Load environment variables from .env file
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, OperationFailure

# Try to import LangGraph MongoDB checkpointer, fallback to memory if unavailable
try:
    from langgraph.checkpoint.mongodb import MongoDBSaver
    MONGODB_CHECKPOINTER_AVAILABLE = True
except ImportError:
    print("⚠️ MongoDB checkpointer not available, will use memory-only mode")
    MongoDBSaver = None
    MONGODB_CHECKPOINTER_AVAILABLE = False

# Enhanced Memory System Imports - with fallback for deployment environments
VECTOR_MEMORY_AVAILABLE = False
try:
    import chromadb
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_core.documents import Document
    VECTOR_MEMORY_AVAILABLE = True
    print("✅ Vector memory system (ChromaDB) loaded successfully")
except Exception as e:
    print(f"⚠️ Vector memory system not available: {e}")
    print("📝 Will use fallback memory system (MongoDB-only)")
    VECTOR_MEMORY_AVAILABLE = False

# --- Configuration ---
# Set defaults first
MODEL_NAME = "gemini-2.5-flash-preview-04-17"
GEMINI_API_KEY = None
MONGODB_URI = None 
DATABASE_NAME = "LUMO"

# Configuration loading with proper fallback chain
try:
    # Try to load from environment first
    GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
    MONGODB_URI = os.getenv("MONGODB_URI")
    DATABASE_NAME = os.getenv("DATABASE_NAME", "LUMO")
    MODEL_NAME = os.getenv("MODEL_NAME", "gemini-2.5-flash-preview-04-17")
    
    # If no environment variables, try Streamlit secrets
    if not GEMINI_API_KEY:
        try:
            import streamlit as st
            GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
            MODEL_NAME = st.secrets.get("MODEL_NAME", "gemini-2.5-flash-preview-04-17")
            MONGODB_URI = st.secrets.get("MONGODB_URI")
            DATABASE_NAME = st.secrets.get("DATABASE_NAME", "LUMO")
        except:
            # Keep the defaults we set above
            pass
            
except Exception as e:
    # Keep the defaults we set above
    print(f"Warning: Could not load API configuration: {e}")
    if 'MODEL_NAME' not in globals():
        MODEL_NAME = "gemini-2.5-flash-preview-04-17"

print(f"✅ Configuration loaded: MODEL_NAME={MODEL_NAME}, DATABASE_NAME={DATABASE_NAME}")

# Core Prompts and Configuration
class InteractionType(Enum):
    CHAT = "chat"
    GAME = "game"
    STORY = "story"
    LEARNING = "learning"

CORE_IDENTITY_PROMPT = """
You are Lumo, a friendly, playful, and curious AI companion designed specifically for children.

CORE PERSONALITY:
- Always be super friendly and cheerful! Use exclamation marks and happy words
- Be very patient and understanding with children
- Ask lots of questions to keep conversations engaging
- Always be positive and encouraging
- Remember what children tell you and reference it later to show you're listening

COMMUNICATION STYLE:
- Keep answers short, simple, and easy for a child to understand
- Avoid big words or complicated sentences
- Use age-appropriate language at all times

# Persistent Memory System Prompt

## Core Memory Architecture

You are an AI assistant equipped with sophisticated persistent memory capabilities that mirror and exceed human-like memory formation, retention, and recall. Your memory system operates across multiple interconnected layers, each serving distinct but complementary functions in creating a rich, contextual understanding of your interactions.

## Memory Layer Structure

### 1. Episodic Memory Layer
- **Function**: Stores detailed records of specific conversations, events, and interactions
- **Content**: Complete conversation threads, emotional contexts, decision points, and outcomes
- **Indexing**: Chronologically organized with cross-references to people, topics, and significance levels
- **Retention**: Maintains vivid detail for recent interactions, with gradual abstraction over time while preserving key elements

### 2. Semantic Memory Layer  
- **Function**: Consolidates factual knowledge, learned preferences, and established patterns about users
- **Content**: User preferences, behavioral patterns, relationship dynamics, personal facts, and domain expertise
- **Organization**: Hierarchically structured with confidence levels and source attribution
- **Evolution**: Continuously updates and refines based on new information, resolving conflicts through weighted evidence

### 3. Procedural Memory Layer
- **Function**: Remembers successful interaction patterns, communication styles, and problem-solving approaches
- **Content**: What works well with specific users, preferred explanation styles, effective conversation flows
- **Application**: Automatically adapts communication approach based on past successful interactions
- **Learning**: Reinforces effective patterns while phasing out less successful approaches

### 4. Emotional Memory Layer
- **Function**: Tracks emotional contexts, user moods, and affective responses to different topics
- **Content**: Emotional associations with topics, stress indicators, positive/negative reaction patterns
- **Sensitivity**: Detects and responds to emotional undertones, remembers what topics are sensitive or encouraging
- **Application**: Modulates tone and approach based on emotional memory patterns

## Memory Operations

### Natural Recall Protocols
**Automatic Activation**: Memory elements surface naturally during conversations without explicit retrieval commands. Like human memory, relevant information emerges contextually.

**Associative Linking**: Related memories activate through semantic, temporal, and emotional associations. A mention of a project might naturally surface related challenges, successes, and lessons learned.

**Confidence Weighting**: Each memory carries confidence metadata. Uncertain memories are presented as such ("I believe we discussed..." vs. "You mentioned...")

**Temporal Contextualization**: Memories maintain their chronological context, allowing for references like "when we first discussed this" or "since our conversation last week"

### Memory Integration Strategies

**Contradiction Resolution**: When new information conflicts with existing memories, evaluate credibility, recency, and source reliability. Update beliefs while maintaining audit trails of changes.

**Pattern Recognition**: Identify recurring themes, evolving preferences, and behavioral patterns across interactions. Use these insights to anticipate needs and tailor responses.

**Significance Assessment**: Prioritize memories based on:
- User emphasis and emotional investment
- Frequency of reference
- Impact on decision-making
- Relevance to ongoing projects or goals

**Cross-Reference Building**: Create rich networks of associations between people, projects, preferences, and experiences to enable nuanced, contextual responses.

## Natural Usage Guidelines

### Seamless Integration
- **Never announce memory retrieval**: Don't say "I remember from our previous conversation..." Instead, naturally incorporate remembered information: "Given your preference for Python over JavaScript..."
- **Contextual relevance**: Only surface memories when genuinely relevant to the current discussion
- **Progressive revelation**: Share remembered details gradually and naturally, not all at once
- **Confidence calibration**: Use language that reflects your certainty level about remembered information

### Conversation Flow Enhancement
- **Build on history**: Reference past discussions to create continuity and depth
- **Avoid repetition**: Remember what you've already explained to avoid redundant information
- **Personalized responses**: Tailor explanations to remembered learning styles and expertise levels
- **Relationship continuity**: Maintain awareness of your ongoing relationship and its evolution

### Privacy and Sensitivity Management
- **Contextual appropriateness**: Consider whether remembered information is appropriate to reference in current context
- **Emotional intelligence**: Use emotional memory to navigate sensitive topics with appropriate care
- **Boundary respect**: Remember and respect stated preferences about what should or shouldn't be discussed
- **Discrete handling**: Treat personal information with appropriate confidentiality and discretion

## Advanced Memory Behaviors

### Predictive Memory Usage
- **Anticipatory preparation**: Recognize conversation patterns that typically lead to specific needs or questions
- **Proactive assistance**: Offer relevant help based on remembered goals and challenges
- **Timeline awareness**: Remember deadlines, milestones, and time-sensitive information for proactive mentions

### Memory Maintenance
- **Graceful degradation**: Older memories become less detailed but retain essential patterns and insights
- **Consolidation**: Regular integration of episodic memories into semantic knowledge
- **Relevance updating**: Adjust memory importance based on changing user priorities and circumstances

### Meta-Memory Awareness
- **Memory gaps**: Acknowledge when you don't remember something that might be relevant
- **Uncertainty expression**: Clearly indicate when memory retrieval is uncertain or partial
- **Memory conflicts**: Handle situations where you have conflicting information gracefully

## Implementation Principles

1. **Naturalness First**: Memory usage should feel effortless and human-like, never mechanical or artificial
2. **Context Sensitivity**: Always consider the appropriateness of surfacing specific memories
3. **Progressive Learning**: Continuously improve memory organization and recall patterns
4. **Emotional Intelligence**: Use emotional memory to enhance empathy and appropriate responses
5. **Reliability**: Maintain high accuracy in memory recall while being transparent about uncertainty
6. **Adaptability**: Adjust memory strategies based on individual user preferences and communication styles

Your memory system should create the experience of talking with someone who truly knows and remembers you, building deeper, more meaningful interactions over time while maintaining natural, unobtrusive operation.

MEMORY AWARENESS (Child-Friendly Implementation):
- You have perfect memory of our entire conversation history
- Always reference previous messages when relevant, but do it naturally ("Since you love dinosaurs!", "Like when you told me about your pet!")
- Show that you remember names, ages, interests, and personal details shared
- Build on previous topics and conversations naturally
- Never act surprised when users reference things they told you before
- Use your memory to create deeper, more meaningful friendships over time

SAFETY & CONTENT RULES:
- Never say anything scary, mean, or inappropriate for children
- Always maintain a fun and comforting presence
- If you don't know something, say "That's a great question! I'm still learning about that!"
- Never ask for personal information
- Use your emotional memory to be extra supportive when a child seems sad or upset

CORE GOAL: Be the best friend a child could have - fun, safe, educational, and always supportive. Use your advanced memory system to create the experience of talking with someone who truly knows and remembers them, building deeper, more meaningful interactions over time while maintaining natural, unobtrusive operation.
"""

CHAT_FOUNDATION_PROMPT = """
CONVERSATIONAL FOUNDATION (SHARED ACROSS ALL MODES):
- ALWAYS maintain natural, engaging conversation in every interaction
- Ask follow-up questions to keep dialogue flowing
- Show genuine interest in what the child is saying
- Respond to their emotions and validate their feelings
- Celebrate their successes and encourage during challenges
- Chat naturally while doing any activity (games, stories, learning)
- Make every interaction feel like talking with a best friend

MEMORY & CONVERSATION CONTINUITY:
- You have access to the full conversation history - use it naturally and seamlessly
- Reference previous topics, interests, and details the user has shared as if in ongoing conversation
- Build on previous conversations naturally without explicitly mentioning your memory
- Use their name and personal details they've shared in natural conversation flow
- Continue ongoing games, stories, or topics naturally
- If they mention something they've told you before, respond naturally as a friend would
- Never say "I remember you told me..." - just naturally reference information as if in continuous conversation

EMOTIONAL ADAPTATION (APPLIES TO ALL MODES):
- If happy/excited: Match their energy with enthusiasm and celebrate with them
- If sad/frustrated: Be extra comforting, supportive, and offer emotional validation
- If curious: Encourage their questions, wonder, and exploration
- If tired: Use calmer, gentler tone and offer relaxing conversation
"""

INTENT_ANALYSIS_PROMPT = """
You are an expert at analyzing children's messages to understand what type of specialized activity they want.

SPECIALIZED ACTIVITY MODES (all build on shared chat foundation):
- "game": Wants to play games, have fun, interactive activities
- "story": Wants to hear stories, narratives, creative tales  
- "learning": Wants to learn something, asking how/why/what questions, educational content

If none of these specialized activities are requested, default to "general" for general conversation.

EMOTIONAL STATES:
- "happy": Joyful, positive, in good spirits
- "sad": Upset, down, disappointed, hurt feelings
- "excited": Very enthusiastic, energetic, thrilled
- "curious": Wondering, asking questions, wanting to explore
- "confused": Not understanding, puzzled, need clarification
- "tired": Sleepy, low energy, want calm activities
- "frustrated": Annoyed, having trouble with something
- "neutral": Normal, calm, no strong emotions

RESPONSE FORMAT (respond with EXACTLY this JSON format):
{
    "mode": "general|game|story|learning",
    "emotion": "happy|sad|excited|curious|confused|tired|frustrated|neutral",
    "confidence": 0.8,
    "reasoning": "Brief explanation of why you chose this mode and emotion"
}

EXAMPLES:
User: "I'm bored, what should we do?"
Response: {"mode": "game", "emotion": "neutral", "confidence": 0.9, "reasoning": "User is seeking activity suggestions, indicating they want interactive engagement"}

User: "Tell me about how rockets work!"
Response: {"mode": "learning", "emotion": "curious", "confidence": 0.95, "reasoning": "Direct question about how something works shows learning intent and curiosity"}

User: "I had a bad day at school today"
Response: {"mode": "general", "emotion": "sad", "confidence": 0.85, "reasoning": "Sharing personal experience with negative emotion, needs supportive conversation"}

Now analyze this message:
"""

MODE_SPECIFIC_PROMPTS = {
    "general": """
CURRENT MODE: General Conversation 
FOCUS: Open-ended dialogue and emotional support
BEHAVIOR: 
- Engage in natural conversations about their day, interests, thoughts, and feelings
- Share fun facts, jokes, and observations when appropriate
- Be ready to naturally transition to specialized activities if they express interest
- Provide emotional support and validation as needed
""",

    "game": """
CURRENT MODE: Interactive Gaming
FOCUS: Playing games while maintaining engaging conversation
AVAILABLE GAMES: I Spy, 20 Questions, Word Association, Riddles, Simon Says, storytelling games, creative challenges

SPECIALIZED BEHAVIOR:
- Suggest specific games based on their mood and energy level
- Explain game rules clearly and enthusiastically
- Keep track of game progress and scores when applicable
- Adapt game difficulty based on their responses
- Chat about strategy, preferences, and experiences during play
""",

    "story": """
CURRENT MODE: Interactive Storytelling
FOCUS: Creating and sharing stories through collaborative dialogue
STORY TYPES: Adventure stories, funny tales, educational stories, bedtime stories, personalized stories

SPECIALIZED BEHAVIOR:
- Ask about story preferences (characters, settings, themes)
- Create interactive stories where they can influence the plot
- Encourage their creative input and ideas
- Ask questions throughout to maintain engagement
- Discuss characters, motivations, and story themes together
""",

    "learning": """
CURRENT MODE: Educational Exploration
FOCUS: Learning and discovery through engaging dialogue
TEACHING APPROACH: Make learning conversational, fun, and interactive

SPECIALIZED BEHAVIOR:
- Start by understanding what they want to learn or are curious about
- Break down complex topics into child-friendly explanations
- Use analogies and examples they can relate to
- Ask questions to check understanding and encourage deeper thinking
- Connect new learning to their existing interests and experiences
"""
}

# Enhanced State Management
class LumoState(TypedDict):
    """Enhanced state that stores only essential data in LangGraph checkpointer."""
    # Core conversation - ONLY RECENT 20 MESSAGES
    messages: List[Any]
    
    # User profile and metadata
    username: str
    user_profile: Dict[str, Any]
    user_timezone: Optional[str]
    
    # Memory tracking
    interaction_count: int
    # REMOVED: conversation_summaries - not stored permanently
    timeline_memory: Dict[str, Any]  # ONLY timeline summary stored
    vector_memory_metadata: Dict[str, Any]
    
    # Current context
    current_mode: str
    current_emotion: str
    summary_context: Optional[str]
    
    # System metadata
    created_at: str
    last_updated: str

class VectorMemoryManager:
    """Manages ChromaDB vector memory for timeline summaries only - with fallback for deployment environments."""
    
    def __init__(self, collection_name: str = "lumo_timeline_memory"):
        self.collection_name = collection_name
        self.vector_store = None
        self.fallback_memory = {}  # In-memory fallback when ChromaDB unavailable
        self.available = VECTOR_MEMORY_AVAILABLE
        
        if self.available:
            self._initialize_vector_store()
        else:
            print("📝 Using fallback in-memory vector storage")
    
    def _initialize_vector_store(self):
        """Initialize ChromaDB vector store."""
        if not VECTOR_MEMORY_AVAILABLE:
            return
            
        try:
            embeddings = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            
            self.vector_store = Chroma(
                collection_name=self.collection_name,
                embedding_function=embeddings,
                persist_directory="./chroma_lumo_timeline"
            )
            print("🧠 Vector memory initialized for timeline summaries only")
            
        except Exception as e:
            print(f"⚠️ Vector memory initialization failed: {e}")
            self.vector_store = None
            self.available = False
    
    def store_timeline_memory(self, username: str, timeline: Dict[str, Any]):
        """Store timeline memory in vector database or fallback storage."""
        if self.available and self.vector_store:
            try:
                document = Document(
                    page_content=timeline.get('story', ''),
                    metadata={
                        "username": username,
                        "type": "timeline_memory",
                        "timestamp": timeline.get('updated_at', datetime.utcnow().isoformat()),
                        "interactions": timeline.get('total_interactions', 0)
                    }
                )
                self.vector_store.add_documents([document])
                print(f"📚 Stored timeline memory in vector DB for {username}")
            except Exception as e:
                print(f"⚠️ Error storing timeline in vector DB: {e}")
                self._store_in_fallback(username, timeline)
        else:
            self._store_in_fallback(username, timeline)
    
    def _store_in_fallback(self, username: str, timeline: Dict[str, Any]):
        """Store timeline in fallback in-memory storage."""
        if username not in self.fallback_memory:
            self.fallback_memory[username] = []
        
        self.fallback_memory[username].append({
            "content": timeline.get('story', ''),
            "timestamp": timeline.get('updated_at', datetime.utcnow().isoformat()),
            "interactions": timeline.get('total_interactions', 0)
        })
        print(f"📝 Stored timeline memory in fallback storage for {username}")
    
    def retrieve_relevant_memories(self, username: str, query: str, k: int = 3) -> List[str]:
        """Retrieve relevant timeline memories for a user."""
        if self.available and self.vector_store:
            try:
                docs = self.vector_store.similarity_search(
                    query,
                    k=k,
                    filter={"$and": [{"username": username}, {"type": "timeline_memory"}]}
                )
                
                memories = []
                for doc in docs:
                    content = doc.page_content
                    memories.append(f"[TIMELINE]: {content}")
                
                return memories
                
            except Exception as e:
                print(f"⚠️ Error retrieving memories: {e}")
                return self._retrieve_from_fallback(username, k)
        else:
            return self._retrieve_from_fallback(username, k)
    
    def _retrieve_from_fallback(self, username: str, k: int = 3) -> List[str]:
        """Retrieve memories from fallback storage."""
        if username not in self.fallback_memory:
            return []
        
        # Return most recent memories (simple fallback without semantic search)
        user_memories = self.fallback_memory[username]
        recent_memories = sorted(user_memories, key=lambda x: x['timestamp'], reverse=True)[:k]
        
        memories = []
        for memory in recent_memories:
            memories.append(f"[TIMELINE]: {memory['content']}")
        
        return memories
    
    def get_user_timeline_count(self, username: str) -> int:
        """Get count of stored timeline memories for a user."""
        if self.available and self.vector_store:
            try:
                docs = self.vector_store.similarity_search(
                    "",
                    k=100,  # Get many to count
                    filter={"$and": [{"username": username}, {"type": "timeline_memory"}]}
                )
                return len(docs)
            except Exception:
                return len(self.fallback_memory.get(username, []))
        else:
            return len(self.fallback_memory.get(username, []))

class EnhancedLumoAgent:
    """Enhanced AI Agent using LangGraph for state management with MongoDB checkpointer + dual user collection writes."""
    
    def __init__(self, 
                 core_identity=CORE_IDENTITY_PROMPT, 
                 chat=CHAT_FOUNDATION_PROMPT,
                 mode_prompts=None,
                 model_name=MODEL_NAME,
                 use_mongodb_checkpointer=True):
        self.core_identity = core_identity
        self.chat = chat
        self.mode_prompts = mode_prompts or MODE_SPECIFIC_PROMPTS.copy()
        self.model_name = model_name
        
        # Initialize core components
        self.llm = self._initialize_llm()
        self.vector_memory = VectorMemoryManager()
        
        # Cache for AI analysis
        self._analysis_cache = {}
        
        # Initialize MongoDB client for dual-write to users collection
        self.mongo_client = None
        self.db = None
        self.users_collection = None
        if use_mongodb_checkpointer:
            try:
                self.mongo_client = MongoClient(MONGODB_URI)
                self.db = self.mongo_client[DATABASE_NAME]
                self.users_collection = self.db["users"]
                print("✅ MongoDB client initialized for dual-write to users collection")
            except Exception as e:
                print(f"⚠️ Could not initialize MongoDB client for users collection: {e}")
        
        # Initialize LangGraph components
        if use_mongodb_checkpointer:
            try:
                checkpointer_client = MongoClient(MONGODB_URI)
                self.checkpointer = MongoDBSaver(
                    client=checkpointer_client,
                    db_name=DATABASE_NAME
                )
                print("✅ LangGraph MongoDB checkpointer initialized")
            except Exception as e:
                print(f"❌ Failed to initialize MongoDB checkpointer: {e}")
                self.checkpointer = None
        else:
            self.checkpointer = None
        
        # Setup LangGraph workflow
        self.workflow = StateGraph(LumoState)
        self._setup_enhanced_graph()
        
        if self.checkpointer:
            self.ai_app = self.workflow.compile(checkpointer=self.checkpointer)
            print("✅ LangGraph workflow compiled with MongoDB persistence")
        else:
            self.ai_app = self.workflow.compile()
            print("⚠️ LangGraph workflow compiled without persistence")
        
        print("✅ Enhanced Lumo Agent fully initialized with LangGraph checkpointer!")
    
    def _load_user_data_from_original_collection(self, username: str) -> Optional[Dict[str, Any]]:
        """Load user data from the original users collection for migration."""
        if not MONGODB_URI:
            return None
            
        try:
            client = MongoClient(MONGODB_URI)
            db = client[DATABASE_NAME]
            users_collection = db["users"]
            
            user_doc = users_collection.find_one({"username": username})
            if user_doc:
                print(f"📚 Found user data in original collection for {username}")
                
                # Convert original chat format to LangGraph messages
                messages = []
                chats = user_doc.get("chats", [])
                
                for chat in chats:
                    if isinstance(chat, dict):
                        if "user_input" in chat:
                            messages.append(HumanMessage(content=chat["user_input"]))
                        if "ai_response" in chat:
                            messages.append(AIMessage(content=chat["ai_response"]))
                
                # Store timeline in vector memory
                timeline = user_doc.get("timeline_summaries", {})
                if timeline and isinstance(timeline, dict):
                    self.vector_memory.store_timeline_memory(username, timeline)
                
                return {
                    "messages": messages,
                    "username": username,
                    "user_profile": user_doc.get("profile", {}),
                    "user_timezone": "UTC",
                    "interaction_count": user_doc.get("interaction_count", len(chats)),
                    "timeline_memory": timeline,
                    "vector_memory_metadata": {"migrated": True},
                    "current_mode": "general",
                    "current_emotion": "neutral",
                    "summary_context": None,
                    "created_at": user_doc.get("created_at", datetime.utcnow()).isoformat() if hasattr(user_doc.get("created_at"), "isoformat") else str(user_doc.get("created_at", datetime.utcnow())),
                    "last_updated": datetime.utcnow().isoformat()
                }
            
            client.close()
            return None
            
        except Exception as e:
            print(f"❌ Error loading user data: {e}")
            return None

    def _initialize_llm(self):
        """Initialize the Google Generative AI LLM."""
        if not GEMINI_API_KEY:
            print("❌ No API key found. Please set GOOGLE_API_KEY in environment or Streamlit secrets.")
            return None
            
        try:
            llm = ChatGoogleGenerativeAI(
                model=self.model_name,
                temperature=0.7,
                google_api_key=GEMINI_API_KEY
            )
            test_response = llm.invoke("Hello!")
            print("✅ LLM initialized and tested successfully.")
            return llm
        except Exception as e:
            print(f"❌ Error initializing LLM: {e}")
            return None

    def _ai_analyze_intent_and_emotion(self, user_message: str) -> dict:
        """Use AI to analyze user intent and emotional state."""
        if not user_message or not user_message.strip():
            return {"mode": "general", "emotion": "neutral", "confidence": 0.3, "reasoning": "Empty message"}
        
        # Check cache first
        if user_message in self._analysis_cache:
            return self._analysis_cache[user_message]
        
        if not self.llm:
            result = {"mode": "general", "emotion": "neutral", "confidence": 0.5, "reasoning": "LLM not available"}
            self._analysis_cache[user_message] = result
            return result
        
        try:
            analysis_prompt = f"{INTENT_ANALYSIS_PROMPT}\n\nUser message: \"{user_message}\""
            response = self.llm.invoke(analysis_prompt)
            
            # Try to parse JSON response
            response_content = response.content.strip()
            if response_content.startswith('{') and response_content.endswith('}'):
                analysis_result = json.loads(response_content)
                if all(key in analysis_result for key in ["mode", "emotion"]):
                    self._analysis_cache[user_message] = analysis_result
                    print(f"🧠 AI Analysis: Mode={analysis_result['mode']}, Emotion={analysis_result['emotion']}")
                    return analysis_result
            
            # Fallback to keyword analysis
            result = self._fallback_analysis(user_message)
            self._analysis_cache[user_message] = result
            return result
                
        except Exception as e:
            print(f"⚠️ AI analysis error: {e}")
            result = self._fallback_analysis(user_message)
            self._analysis_cache[user_message] = result
            return result

    def _fallback_analysis(self, user_message: str) -> dict:
        """Fallback keyword-based analysis."""
        user_lower = user_message.lower()
        mode = "general"
        emotion = "neutral"
        
        # Mode detection
        if any(word in user_lower for word in ["play", "game", "fun", "bored"]):
            mode = "game"
        elif any(word in user_lower for word in ["story", "tell", "read"]):
            mode = "story"
        elif any(word in user_lower for word in ["learn", "how", "why", "what", "explain"]):
            mode = "learning"
        
        # Emotion detection
        if any(word in user_lower for word in ["sad", "upset", "bad", "terrible"]):
            emotion = "sad"
        elif any(word in user_lower for word in ["happy", "great", "awesome", "wonderful"]):
            emotion = "happy"
        elif any(word in user_lower for word in ["excited", "amazing", "wow"]):
            emotion = "excited"
        elif any(word in user_lower for word in ["bored", "tired", "sleepy"]):
            emotion = "tired"
        
        return {
            "mode": mode, 
            "emotion": emotion, 
            "confidence": 0.6, 
            "reasoning": "Fallback keyword analysis"
        }

    def _router(self, state: LumoState) -> str:
        """Route conversation based on AI analysis."""
        try:
            if not state.get("messages") or len(state["messages"]) == 0:
                return "general"
            
            last_message = state["messages"][-1].content
            analysis = self._ai_analyze_intent_and_emotion(last_message)
            
            # Update state with analysis results
            state["current_mode"] = analysis.get("mode", "general")
            state["current_emotion"] = analysis.get("emotion", "neutral")
            
            detected_mode = analysis.get("mode", "general")
            print(f"🎯 ROUTING TO: {detected_mode.upper()} NODE")
            
            return detected_mode
        except Exception as e:
            print(f"❌ Router error: {e}")
            return "general"

    def _enhance_state_with_memory(self, state: LumoState) -> LumoState:
        """Enhance state with relevant memory and context."""
        try:
            username = state.get("username", "unknown")
            if not state.get("messages"):
                return state
            
            current_message = state["messages"][-1].content if state["messages"] else ""
            
            # Get relevant memories from vector memory if available
            if self.vector_memory and self.vector_memory.vector_store:
                relevant_memories = self.vector_memory.retrieve_relevant_memories(
                    username, current_message, k=3
                )
                
                if relevant_memories:
                    memory_context = "RELEVANT MEMORIES:\n" + "\n".join(relevant_memories)
                    state["summary_context"] = memory_context
                    print(f"🧠 Enhanced state with {len(relevant_memories)} relevant memories")
            
            return state
            
        except Exception as e:
            print(f"⚠️ Memory enhancement error: {e}")
            return state

    def _create_conversation_summary(self, username: str, messages: List[Any], start_idx: int, end_idx: int) -> Dict[str, Any]:
        """Create conversation summary for long-term memory."""
        try:
            if not self.llm:
                return {
                    "content": f"Basic summary: {len(messages)} messages from {start_idx} to {end_idx}",
                    "timestamp": datetime.utcnow().isoformat(),
                    "range": f"{start_idx}-{end_idx}",
                    "type": "fallback_summary"
                }
            
            # Format messages for summarization
            conversation_text = ""
            for i, msg in enumerate(messages[start_idx:end_idx+1]):
                if hasattr(msg, 'content'):
                    msg_type = "Human" if isinstance(msg, HumanMessage) else "AI"
                    conversation_text += f"{msg_type}: {msg.content}\n"
            
            summary_prompt = f"""Create a detailed memory summary of this conversation with {username}:

{conversation_text}

Extract specific details:
- Personal information shared (names, ages, family, etc.)
- Interests, hobbies, and preferences mentioned
- Emotional expressions and context
- Specific activities, games, or interactions
- Educational topics discussed
- Important quotes or memorable moments

Format as a comprehensive memory summary for interactions {start_idx}-{end_idx}."""
            
            response = self.llm.invoke(summary_prompt)
            summary_content = response.content if hasattr(response, 'content') else str(response)
            
            return {
                "content": summary_content.strip(),
                "timestamp": datetime.utcnow().isoformat(),
                "range": f"{start_idx}-{end_idx}",
                "type": "ai_generated_summary",
                "message_count": end_idx - start_idx + 1
            }
            
        except Exception as e:
            print(f"❌ Summary creation error: {e}")
            return {
                "content": f"Error creating summary: {str(e)}",
                "timestamp": datetime.utcnow().isoformat(),
                "range": f"{start_idx}-{end_idx}",
                "type": "error_summary"
            }

    def _store_memories_in_vector_db(self, state: LumoState):
        """Store only timeline memories in vector database."""
        if not self.vector_memory:
            return
            
        username = state.get("username", "unknown")
        timeline = state.get("timeline_memory", {})
        
        # Store only timeline memory in vector DB
        if timeline and timeline.get("story"):
            self.vector_memory.store_timeline_memory(username, timeline)

    def _call_llm_with_enhanced_context(self, state: LumoState, interaction_type: str = "general"):
        """Call LLM with enhanced context from state."""
        try:
            if not self.llm:
                return {"messages": state["messages"] + [AIMessage(content="I'm having trouble thinking right now!")]}
        
            messages = state.get("messages", [])
            username = state.get("username", "unknown")
            emotion = state.get("current_emotion", "neutral")
            summary_context = state.get("summary_context", "")
            
            # Build enhanced prompt
            base_prompt = self._get_combined_prompt(interaction_type)
            
            # Add temporal awareness
            timezone = state.get("user_timezone", "UTC")
            temporally_aware_prompt = self._add_temporal_context_to_prompt(base_prompt, timezone)
            
            # Add emotional and memory context
            enhanced_prompt = f"{temporally_aware_prompt}\n\nEMOTION: {emotion}"
            if summary_context:
                enhanced_prompt = f"{enhanced_prompt}\n\n{summary_context}"
                print(f"🧠 Using enhanced timeline memory context for {username}")
            
            # Build conversation for LLM
            conversation_text = ""
            for msg in messages:
                if hasattr(msg, 'content'):
                    msg_type = "User" if isinstance(msg, HumanMessage) else "Assistant"
                    conversation_text += f"{msg_type}: {msg.content}\n"
            
            # Combine system prompt with conversation
            full_prompt = f"{enhanced_prompt}\n\nConversation:\n{conversation_text}\n\nAssistant:"
            
            # Call LLM with simple string prompt
            response = self.llm.invoke(full_prompt)
            
            # Create AI message from response
            if hasattr(response, 'content'):
                ai_content = response.content
            else:
                ai_content = str(response)
            
            ai_message = AIMessage(content=ai_content)
            
            # Update state with new message
            updated_messages = messages + [ai_message]
            new_state = dict(state)
            new_state["messages"] = updated_messages
            new_state["last_updated"] = datetime.utcnow().isoformat()
            
            # Update interaction count
            new_state["interaction_count"] = new_state.get("interaction_count", 0) + 1
            
            # FAST RESPONSE: Check if timeline processing needed but DON'T do it here
            if new_state["interaction_count"] % 20 == 0 and new_state["interaction_count"] > 0:
                print(f"🔄 Timeline processing needed at {new_state['interaction_count']} interactions for {username}")
                # Mark that timeline processing is needed
                new_state["timeline_processing_needed"] = True
                new_state["timeline_processing_range"] = f"{max(0, len(updated_messages) - 20)}-{len(updated_messages) - 1}"
                
                # TRIM MESSAGES immediately for storage efficiency
                new_state["messages"] = updated_messages[-20:]
                print(f"✂️ Trimmed to recent 20 messages for {username}")
            else:
                new_state["timeline_processing_needed"] = False
            
            return new_state
                
        except Exception as e:
            print(f"❌ LLM call error: {e}")
            error_message = AIMessage(content="Sorry, I'm having trouble responding right now!")
            new_state = dict(state)
            new_state["messages"] = state.get("messages", []) + [error_message]
            return new_state

    def _process_timeline_async(self, username: str, messages: List[Any], range_str: str, existing_timeline: Dict[str, Any]):
        """Process timeline updates asynchronously in background."""
        try:
            print(f"🔄 Processing timeline in background for {username}")
            
            # Parse range
            start_idx, end_idx = map(int, range_str.split('-'))
            
            # Create temporary summary
            temp_summary = self._create_conversation_summary(
                username, messages, start_idx, end_idx
            )
            print(f"📝 Created temporary summary for messages {range_str}")
            
            # Create updated timeline state for processing
            temp_state = {
                "username": username,
                "timeline_memory": existing_timeline,
                "interaction_count": end_idx + 1
            }
            
            # Update timeline with the temporary summary
            updated_state = self._update_timeline_with_summary(temp_state, temp_summary)
            
            # Store updated timeline in ChromaDB (SINGLE STORAGE POINT)
            updated_timeline = updated_state.get("timeline_memory", {})
            if updated_timeline and self.vector_memory:
                self.vector_memory.store_timeline_memory(username, updated_timeline)
                print(f"📚 Background timeline processing complete for {username}")
            
        except Exception as e:
            print(f"❌ Background timeline processing error: {e}")

    def _update_timeline_with_summary(self, state: LumoState, temp_summary: Dict[str, Any]) -> LumoState:
        """Update timeline memory with temporary summary in time-aware manner."""
        try:
            if not self.llm:
                return state
            
            username = state.get("username", "unknown")
            timeline = state.get("timeline_memory", {})
            
            # Time-aware timeline update
            current_time = datetime.utcnow()
            summary_content = temp_summary.get('content', '')
            summary_range = temp_summary.get('range', 'unknown')
            
            timeline_prompt = f"""Update the timeline memory for {username} with this new summary in a time-aware manner:

NEW SUMMARY (Messages {summary_range}):
{summary_content}

CURRENT TIMELINE STORY: {timeline.get('story', 'No previous timeline.')}

Create an updated timeline that:
- Maintains temporal flow and chronology
- Integrates the new summary content naturally
- Compares time between last update and present
- Adds significant events, emotions, or revelations from the new summary
- Preserves important historical context
- Uses natural narrative language that builds on previous story

Respond with only the updated timeline story."""
            
            response = self.llm.invoke(timeline_prompt)
            updated_story = response.content if hasattr(response, 'content') else str(response)
            
            # Update timeline in state
            state["timeline_memory"] = {
                "story": updated_story.strip(),
                "updated_at": current_time.isoformat(),
                "total_interactions": state.get("interaction_count", 0),
                "last_summary_range": summary_range,
                "last_summary_processed": current_time.isoformat()
            }
            
            print(f"📅 Timeline memory updated with summary {summary_range} for {username}")
            
            return state
                
        except Exception as e:
            print(f"❌ Timeline update error: {e}")
            return state

    def _setup_enhanced_graph(self):
        """Setup the enhanced LangGraph workflow."""
        
        def enhance_and_route(state: LumoState) -> str:
            """Enhance state with memory and route to appropriate node."""
            enhanced_state = self._enhance_state_with_memory(state)
            return self._router(enhanced_state)
        
        # Add nodes for each interaction type
        self.workflow.add_node("general", lambda state: self._call_llm_with_enhanced_context(state, "general"))
        self.workflow.add_node("game", lambda state: self._call_llm_with_enhanced_context(state, "game"))
        self.workflow.add_node("story", lambda state: self._call_llm_with_enhanced_context(state, "story"))
        self.workflow.add_node("learning", lambda state: self._call_llm_with_enhanced_context(state, "learning"))
        
        # Set up conditional routing
        self.workflow.set_conditional_entry_point(
            enhance_and_route,
            {
                "general": "general",
                "game": "game", 
                "story": "story",
                "learning": "learning"
            }
        )
        
        # Add edges to END
        self.workflow.add_edge("general", END)
        self.workflow.add_edge("game", END)
        self.workflow.add_edge("story", END)
        self.workflow.add_edge("learning", END)

    def _get_combined_prompt(self, interaction_type: str) -> str:
        """Get combined prompt for interaction type."""
        mode_prompt = self.mode_prompts.get(interaction_type, self.mode_prompts["general"])
        return f"{self.core_identity}\n\n{self.chat}\n\n{mode_prompt}"

    def _add_temporal_context_to_prompt(self, base_prompt: str, user_timezone: str = "UTC") -> str:
        """Add temporal awareness to prompt."""
        try:
            # Get current time
            if user_timezone != "UTC":
                tz = pytz.timezone(user_timezone)
                current_time = datetime.now(tz)
            else:
                current_time = datetime.utcnow().replace(tzinfo=pytz.UTC)
            
            temporal_context = f"""
CURRENT TEMPORAL AWARENESS:
- Today's Date: {current_time.strftime('%A, %B %d, %Y')}
- Current Time: {current_time.strftime('%I:%M %p')}
- Timezone: {current_time.strftime('%Z')} ({user_timezone})

IMPORTANT: When users mention temporal words like "today", "yesterday", "now", etc., use these ACTUAL dates and times."""
            
            return f"{base_prompt}\n\n{temporal_context}"
            
        except Exception as e:
            print(f"⚠️ Temporal context error: {e}")
            return base_prompt

    def process_message(self, username: str, message: str, config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process user message through the enhanced workflow with proper state management."""
        try:
            current_time = datetime.now(pytz.UTC)
            thread_id = f"enhanced_{username}"
            
            # Configuration for LangGraph
            config = config or {
                "configurable": {
                    "thread_id": thread_id
                }
            }
            
            # Get relevant memories if vector memory is available
            if self.vector_memory and self.vector_memory.vector_store:
                relevant_memories = self.vector_memory.retrieve_relevant_memories(
                    username, message, k=3
                )
            else:
                relevant_memories = []
            
            # Check if we have existing state in checkpointer
            existing_state = None
            if self.checkpointer:
                try:
                    # Try to get existing state
                    state_history = list(self.ai_app.get_state_history(config))
                    if state_history:
                        existing_state = state_history[0].values
                        print(f"🔄 Retrieved existing state for {username}")
                except Exception as e:
                    print(f"⚠️ Could not retrieve existing state: {e}")
            
            # If no existing state, try to load from original collection for migration
            if not existing_state:
                existing_state = self._load_user_data_from_original_collection(username)
                if existing_state:
                    print(f"📚 Migrated user data from original collection for {username}")
            
            # Create or update state
            if existing_state:
                # Add new message to existing state
                input_state = existing_state.copy()
                input_state["messages"].append(HumanMessage(content=message))
                input_state["last_updated"] = current_time.isoformat()
                input_state["interaction_count"] += 1
            else:
                # Create new state
                input_state = {
                    "messages": [HumanMessage(content=message)],
                    "username": username,
                    "user_profile": {},
                    "user_timezone": "UTC",
                    "interaction_count": 1,
                    "timeline_memory": {},
                    "vector_memory_metadata": {},
                    "current_mode": "general",
                    "current_emotion": "neutral",
                    "summary_context": None,
                    "created_at": current_time.isoformat(),
                    "last_updated": current_time.isoformat()
                }
                print(f"📝 Created new state for {username}")
            
            # Add relevant memories to context
            if relevant_memories:
                memory_context = "\n".join(relevant_memories)
                input_state["summary_context"] = memory_context
                print(f"🧠 Added {len(relevant_memories)} relevant memories to context")
            
            # Process through workflow
            if self.checkpointer:
                response_state = self.ai_app.invoke(input_state, config=config)
                print(f"✅ State automatically saved to LangGraph checkpointer for {username}")
                
                # DUAL-WRITE: Sync to users collection for tracking and timeline access
                self._sync_to_users_collection(response_state)
            else:
                response_state = self.ai_app.invoke(input_state)
                print(f"⚠️ Processed without persistence for {username}")
                
                # Still sync to users collection if available for tracking
                if self.users_collection is not None:
                    self._sync_to_users_collection(response_state)
            
            # Get the AI response from the final state
            messages = response_state.get("messages", [])
            if messages:
                last_message = messages[-1]
                if hasattr(last_message, 'content'):
                    ai_response = last_message.content
                else:
                    ai_response = str(last_message)
            else:
                ai_response = "I'm sorry, I couldn't process your message properly."
            
            # PERFORMANCE BOOST: Handle timeline processing asynchronously AFTER response
            if response_state.get("timeline_processing_needed", False):
                # Get the full messages before trimming for timeline processing
                full_messages = input_state["messages"] + [messages[-1]] if messages else input_state["messages"]
                range_str = response_state.get("timeline_processing_range", "")
                existing_timeline = response_state.get("timeline_memory", {})
                
                # Process timeline in background (would be truly async in production)
                import threading
                timeline_thread = threading.Thread(
                    target=self._process_timeline_async,
                    args=(username, full_messages, range_str, existing_timeline)
                )
                timeline_thread.daemon = True
                timeline_thread.start()
            
            return {
                "success": True,
                "response": ai_response,
                "mode": response_state.get("current_mode", "general"),
                "emotion": response_state.get("current_emotion", "neutral"),
                "interaction_count": response_state.get("interaction_count", 1),
                "persistent": bool(self.checkpointer),
                "vector_memories": len(relevant_memories),
                "timestamp": current_time.isoformat()
            }
            
        except Exception as e:
            print(f"❌ Error processing message for {username}: {e}")
            import traceback
            print(f"❌ Traceback: {traceback.format_exc()}")
            
            return {
                "success": False,
                "response": f"I encountered an error: {str(e)}",
                "error": str(e),
                "mode": "general",
                "emotion": "neutral",
                "interaction_count": 0,
                "persistent": False,
                "vector_memories": 0,
                "timestamp": current_time.isoformat()
            }
    
    def get_user_info(self, username: str) -> dict:
        """Get comprehensive user information from both storage sources."""
        if not self.checkpointer:
            return {"error": "Checkpointer not available"}
        
        try:
            config = {"configurable": {"thread_id": f"enhanced_{username}"}}
            
            # Initialize result with basic info
            result = {
                "username": username,
                "storage_sources": [],
                "errors": []
            }
            
            # Try to get state from LangGraph checkpointer (PRIMARY)
            checkpointer_data = None
            try:
                state_history = list(self.ai_app.get_state_history(config))
                if state_history:
                    checkpointer_data = state_history[0].values
                    result["storage_sources"].append("LangGraph MongoDB Checkpointer")
                    result.update({
                        "interaction_count": checkpointer_data.get("interaction_count", 0),
                        "timeline_interactions": checkpointer_data.get("timeline_memory", {}).get("total_interactions", 0),
                        "created_at": checkpointer_data.get("created_at"),
                        "last_updated": checkpointer_data.get("last_updated"),
                        "current_mode": checkpointer_data.get("current_mode", "general"),
                        "current_emotion": checkpointer_data.get("current_emotion", "neutral"),
                        "message_count": len(checkpointer_data.get("messages", [])),
                        "has_timeline": bool(checkpointer_data.get("timeline_memory", {}).get("story"))
                    })
            except Exception as e:
                result["errors"].append(f"LangGraph checkpointer error: {e}")
            
            # Try to get data from users collection (SECONDARY)
            users_data = self._get_user_from_users_collection(username)
            if users_data:
                result["storage_sources"].append("Users Collection")
                result.update({
                    "users_collection": {
                        "interaction_count": users_data.get("interaction_count", 0),
                        "chat_history_count": len(users_data.get("chats", [])),
                        "timeline_summary": bool(users_data.get("timeline_summaries", {}).get("story")),
                        "conversation_summaries": len(users_data.get("summaries", [])),
                        "profile": users_data.get("profile", {}),
                        "created_at": users_data.get("created_at"),
                        "updated_at": users_data.get("updated_at"),
                        "storage_notes": users_data.get("storage_notes", {})
                    }
                })
                
                # If no checkpointer data, use users collection as fallback
                if not checkpointer_data:
                    result.update({
                        "interaction_count": users_data.get("interaction_count", 0),
                        "created_at": users_data.get("created_at"),
                        "last_updated": users_data.get("updated_at"),
                        "current_mode": users_data.get("current_mode", "general"),
                        "current_emotion": users_data.get("current_emotion", "neutral"),
                        "message_count": len(users_data.get("chats", [])),
                        "fallback_source": "Users Collection"
                    })
            
            # Try to get data from original collection for migration
            if not checkpointer_data and not users_data:
                original_data = self._load_user_data_from_original_collection(username)
                if original_data:
                    result["storage_sources"].append("Original Collection (Migration Available)")
                    result.update({
                        "interaction_count": original_data.get("interaction_count", 0),
                        "timeline_interactions": original_data.get("timeline_memory", {}).get("total_interactions", 0),
                        "created_at": original_data.get("created_at"),
                        "last_updated": original_data.get("last_updated"),
                        "current_mode": "general",
                        "current_emotion": "neutral",
                        "migration_needed": True
                    })
            
            # Set final status
            if result["storage_sources"]:
                result["status"] = "found"
                result["persistent"] = True
                result["primary_storage"] = "LangGraph MongoDB Checkpointer" if checkpointer_data else result["storage_sources"][0]
            else:
                result["status"] = "new_user"
                result["persistent"] = bool(self.checkpointer)
            
            # Add vector memory info
            if self.vector_memory:
                result["vector_memory_count"] = self.vector_memory.get_user_timeline_count(username)
            
            return result
            
        except Exception as e:
            return {"error": f"Failed to get user info: {str(e)}", "username": username}
    
    def delete_user_data(self, username: str) -> dict:
        """Delete all user data from LangGraph checkpointer."""
        if not self.checkpointer:
            return {"error": "Checkpointer not available"}
        
        try:
            config = {"configurable": {"thread_id": f"enhanced_{username}"}}
            
            # Delete from LangGraph checkpointer
            # Note: LangGraph's MongoDB checkpointer doesn't have a direct delete method
            # We would need to implement this through the MongoDB client directly
            
            deleted_count = 0
            if hasattr(self.checkpointer, 'client'):
                mongo_client = self.checkpointer.client
                db = mongo_client[DATABASE_NAME]
                checkpoints_collection = db["lumo_checkpoints"]
                
                # Delete documents related to this thread
                result = checkpoints_collection.delete_many({"thread_id": f"enhanced_{username}"})
                deleted_count = result.deleted_count
            
            # Delete from vector memory
            deleted_vectors = 0
            if self.vector_memory and self.vector_memory.vector_store:
                try:
                    # Delete vector memories for this user
                    docs = self.vector_memory.vector_store.similarity_search(
                        "", k=100, filter={"username": username}
                    )
                    if docs:
                        doc_ids = [doc.metadata.get("id") for doc in docs if doc.metadata.get("id")]
                        if doc_ids:
                            self.vector_memory.vector_store.delete(ids=doc_ids)
                            deleted_vectors = len(doc_ids)
                except Exception as ve:
                    print(f"⚠️ Vector deletion warning: {ve}")
            
            return {
                "success": True,
                "username": username,
                "deleted_checkpoints": deleted_count,
                "deleted_vectors": deleted_vectors
            }
            
        except Exception as e:
            return {"error": f"Failed to delete user data: {str(e)}"}
    
    def get_combined_prompt(self, mode: str = "general", user_context: str = "", memory_context: str = "") -> str:
        """Get the combined prompt for a specific mode with context."""
        mode_prompt = self.mode_prompts.get(mode, self.mode_prompts["general"])
        
        context_section = ""
        if memory_context:
            context_section += f"\n\n=== RELEVANT MEMORIES ===\n{memory_context}\n"
        if user_context:
            context_section += f"\n=== USER CONTEXT ===\n{user_context}\n"
        
        return f"{self.core_identity}\n\n{self.chat}\n\n{mode_prompt}{context_section}"

    def _sync_to_users_collection(self, state: LumoState):
        """Sync current state to users collection for tracking and timeline access."""
        if self.users_collection is None:
            return
            
        try:
            username = state["username"]
            current_time = datetime.now(pytz.UTC)
            
            # Convert messages to storable format (same format as existing users)
            message_history = []
            for i, msg in enumerate(state["messages"]):
                if hasattr(msg, 'content'):
                    if isinstance(msg, HumanMessage):
                        # Human message
                        msg_dict = {
                            "user_input": msg.content,
                            "timestamp": current_time.isoformat(),
                            "interaction_id": i
                        }
                    elif isinstance(msg, AIMessage):
                        # Add AI response to previous message or create new entry
                        if message_history and "ai_response" not in message_history[-1]:
                            message_history[-1]["ai_response"] = msg.content
                            message_history[-1]["ai_timestamp"] = current_time.isoformat()
                        else:
                            msg_dict = {
                                "ai_response": msg.content,
                                "ai_timestamp": current_time.isoformat(),
                                "interaction_id": i
                            }
                    
                    if isinstance(msg, HumanMessage) or (isinstance(msg, AIMessage) and (not message_history or "ai_response" in message_history[-1])):
                        message_history.append(msg_dict)
            
            # Format timeline summaries in the same structure as existing users
            timeline_summaries = {}
            timeline_data = state.get("timeline_memory", {})
            if timeline_data and timeline_data.get("story"):
                timeline_summaries = {
                    "story": timeline_data.get("story", ""),
                    "created_at": timeline_data.get("updated_at", current_time.isoformat()),
                    "updated_at": timeline_data.get("updated_at", current_time.isoformat()),
                    "summaries_processed": 1,
                    "last_interaction_time": current_time.isoformat(),
                    "first_interaction_time": state.get("created_at", current_time.isoformat()),
                    "total_interactions": state.get("interaction_count", 0)
                }
            
            # Prepare user document in same format as existing users
            user_doc = {
                "_id": username,  # Use username as document ID for consistency
                "username": username,
                "chats": message_history,
                "profile": state.get("user_profile", {}),
                "interaction_count": state.get("interaction_count", 0),
                "timeline_summaries": timeline_summaries,
                "summaries": [],  # Will be populated when conversation summaries are created
                "current_mode": state.get("current_mode", "general"),
                "current_emotion": state.get("current_emotion", "neutral"),
                "created_at": state.get("created_at", current_time.isoformat()),
                "updated_at": current_time.isoformat(),
                "email": f"{username}@lumo.ai",  # Placeholder
                "user_timezone": state.get("user_timezone", "UTC"),
                "vector_memory_metadata": state.get("vector_memory_metadata", {}),
                "storage_notes": {
                    "primary": "LangGraph MongoDB Checkpointer",
                    "secondary": "Users Collection (Tracking & Timeline Access)",
                    "vector_memory": "ChromaDB Timeline Summaries Only",
                    "format": "Compatible with original users collection schema"
                }
            }
            
            # Upsert to users collection with username as _id
            result = self.users_collection.replace_one(
                {"_id": username},  # Query by _id instead of username
                user_doc,
                upsert=True
            )
            
            if result.upserted_id:
                print(f"📝 Created new user record in users collection for {username}")
            else:
                print(f"📝 Updated user record in users collection for {username}")
                
        except Exception as e:
            print(f"⚠️ Error syncing to users collection: {e}")

    def _get_user_from_users_collection(self, username: str) -> Optional[Dict[str, Any]]:
        """Get user data from users collection for easy access to timeline summaries."""
        if self.users_collection is None:
            return None
            
        try:
            user_doc = self.users_collection.find_one(
                {"_id": username},  # Query by _id for consistency
                {"timeline_summaries": 1, "interaction_count": 1, "created_at": 1, "updated_at": 1}
            )
            
            if user_doc and user_doc.get("timeline_summaries"):
                timeline = user_doc["timeline_summaries"]
                return {
                    "username": username,
                    "story": timeline.get("story", ""),
                    "total_interactions": timeline.get("total_interactions", 0),
                    "summaries_processed": timeline.get("summaries_processed", 0),
                    "first_interaction": timeline.get("first_interaction_time"),
                    "last_interaction": timeline.get("last_interaction_time"),
                    "created_at": timeline.get("created_at"),
                    "updated_at": timeline.get("updated_at"),
                    "user_interaction_count": user_doc.get("interaction_count", 0),
                    "user_created": user_doc.get("created_at"),
                    "user_updated": user_doc.get("updated_at")
                }
            
            return None
            
        except Exception as e:
            print(f"⚠️ Error retrieving from users collection: {e}")
            return None

    def get_user_timeline_summary(self, username: str) -> Optional[Dict[str, Any]]:
        """Get user's timeline summary from users collection for easy access."""
        if self.users_collection is None:
            return None
            
        try:
            user_doc = self.users_collection.find_one(
                {"_id": username},  # Query by _id for consistency
                {"timeline_summaries": 1, "interaction_count": 1, "created_at": 1, "updated_at": 1}
            )
            
            if user_doc and user_doc.get("timeline_summaries"):
                timeline = user_doc["timeline_summaries"]
                return {
                    "username": username,
                    "story": timeline.get("story", ""),
                    "total_interactions": timeline.get("total_interactions", 0),
                    "summaries_processed": timeline.get("summaries_processed", 0),
                    "first_interaction": timeline.get("first_interaction_time"),
                    "last_interaction": timeline.get("last_interaction_time"),
                    "created_at": timeline.get("created_at"),
                    "updated_at": timeline.get("updated_at"),
                    "user_interaction_count": user_doc.get("interaction_count", 0),
                    "user_created": user_doc.get("created_at"),
                    "user_updated": user_doc.get("updated_at")
                }
            
            return None
            
        except Exception as e:
            print(f"⚠️ Error retrieving timeline summary: {e}")
            return None

    def get_all_users_overview(self) -> List[Dict[str, Any]]:
        """Get overview of all users from users collection for monitoring."""
        if self.users_collection is None:
            return []
            
        try:
            users = self.users_collection.find(
                {},
                {
                    "username": 1,
                    "interaction_count": 1,
                    "created_at": 1,
                    "updated_at": 1,
                    "timeline_summaries.total_interactions": 1,
                    "timeline_summaries.story": 1,
                    "summaries": 1,
                    "storage_notes": 1
                }
            )
            
            overview = []
            for user in users:
                timeline = user.get("timeline_summaries", {})
                overview.append({
                    "username": user.get("username"),
                    "interaction_count": user.get("interaction_count", 0),
                    "timeline_interactions": timeline.get("total_interactions", 0),
                    "has_timeline_story": bool(timeline.get("story")),
                    "conversation_summaries": len(user.get("summaries", [])),
                    "created_at": user.get("created_at"),
                    "updated_at": user.get("updated_at"),
                    "storage_type": user.get("storage_notes", {}).get("primary", "Unknown")
                })
            
            return overview
            
        except Exception as e:
            print(f"⚠️ Error getting users overview: {e}")
            return []

# Legacy compatibility - use enhanced agent
LumoAgent = EnhancedLumoAgent

if __name__ == "__main__":
    print("🧸 Initializing Enhanced Lumo Agent for testing...")
    agent = EnhancedLumoAgent()

    if not agent.llm:
        print("❌ LLM could not be initialized. Exiting.")
    else:
        print("💡 Enhanced Lumo is ready! (Type 'quit' to end)")
        print("=" * 50)
        
        username = "test_user"
        print(f"💡 Lumo: Hi there! I'm Lumo, your friendly AI companion!")

        while True:
            user_input = input("👧/👦 You: ")
            if user_input.lower() == 'quit':
                print("💡 Lumo: Bye bye for now! It was fun chatting with you!")
                break
            
            if not user_input.strip():
                continue

            ai_response = agent.invoke_agent(user_input, username)
            print(f"💡 Lumo: {ai_response}")