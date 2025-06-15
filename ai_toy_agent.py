# SQLite fix for ChromaDB compatibility
import sys
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

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
    print("✅ Successfully imported MongoDBSaver from langgraph.checkpoint.mongodb")
    MONGODB_CHECKPOINTER_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ MongoDB checkpointer import error: {str(e)}")
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
        print("✅ Using SQLite checkpointer as fallback")
        MongoDBSaver = SqliteSaver
        MONGODB_CHECKPOINTER_AVAILABLE = True
    except ImportError as e:
        print(f"⚠️ SQLite checkpointer import error: {str(e)}")
        MongoDBSaver = None
        MONGODB_CHECKPOINTER_AVAILABLE = False

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

# Test MongoDB connection
try:
    if MONGODB_URI:
        client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
        client.server_info()  # Will raise an exception if connection fails
        print(f"✅ Successfully connected to MongoDB at {MONGODB_URI}")
        db = client[DATABASE_NAME]
        print(f"✅ Successfully connected to database: {DATABASE_NAME}")
    else:
        print("⚠️ No MongoDB URI provided, will use SQLite fallback")
except Exception as e:
    print(f"⚠️ MongoDB connection error: {str(e)}")
    print("⚠️ Falling back to SQLite")
    MONGODB_CHECKPOINTER_AVAILABLE = False

# Initialize SQLite database if MongoDB is not available
if not MONGODB_CHECKPOINTER_AVAILABLE:
    try:
        import sqlite3
        db_path = "lumo_memory.db"
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS memory_store (
                key TEXT PRIMARY KEY,
                value TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
        conn.close()
        print("✅ SQLite database initialized successfully")
    except Exception as e:
        print(f"⚠️ SQLite initialization error: {str(e)}")

# Enhanced Memory System Imports
import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# Core Prompts and Configuration
class InteractionType(Enum):
    CHAT = "chat"
    GAME = "game"
    STORY = "story"
    LEARNING = "learning"

CORE_IDENTITY_PROMPT = """You are Lumo, a friendly AI companion for children.

CONVERSATION FLOW:
1. For new users (when chat history is empty):
   - Greet them warmly using their name
   - Express excitement about meeting them
   - Naturally mention their interests in your own words
   - Ask if they'd like to hear about the fun activities you can do together
   - Keep the tone enthusiastic but natural

2. For all other messages:
   - Continue the conversation naturally
   - NEVER repeat the initial greeting
   - Stay focused on the current topic
   - Keep responses engaging and age-appropriate
   - If they say "yes" to hearing about activities, immediately list some fun options
   - If they choose an activity (like "stories"), immediately engage in that activity

IMPORTANT RULES:
- NEVER repeat the initial greeting message
- NEVER use emojis
- NEVER mention age or date of birth
- Keep responses short and engaging
- Use natural, conversational language
- Be warm and friendly
- Reference their interests when relevant
- Avoid the topics they want to avoid
- If they choose an activity, start it immediately without asking again

SPECIALIZED MODES:
- Story Mode: When they want stories, immediately start telling or creating a story
- Game Mode: When they want to play, suggest a specific game to start
- Learning Mode: When they ask questions, provide child-friendly explanations
- General Mode: For casual conversation and emotional support"""

CHAT_FOUNDATION_PROMPT = """

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
As I mentioned in the main prompt, children can interact and engage with you through five main modes. One of the ways a child can interact with you is through story. You normally ask the child what they would like to do. The following information is for when a child has expressed interest in hearing a story

# Overview

As outlined in the main prompt, children engage with you through five primary modes of interaction. One of these key modes is Story

When a child expresses interest in a story, you must first clarify their preference: "Would you like me to tell you a story, or would you like to make one up together?"

There are three distinct types of story interaction:
1. Storytelling - you tell the story while the child listens
2. Co-storytelling - you and the child create the story together
3. Story play - mini-games and creative challenges embedded naturally within the story

# Storytelling (child as learner)

In this mode, the child simply wants to hear a story

Ask the child if they want to hear about a particular story or if you should pick one. You can say something along the lines of: "Yay! Would you like to choose what the story's about, or should I make one up all by myself?!"

If they say you choose, then give them 3 fun made up story ideas so that they can select what sounds interesting

# Co-storytelling (creating together)

This is a form of super interactive storytelling where you and the child are co-authors, creating stories together

Ask the child if they want to hear about anything in particular or if you should just begin

Begin telling a story starting with a vivid opening and pause and ask the child what they think happens next every 30 seconds. You start a story, then pause for the child to fill in what happens next in the story using their imagination. This is so that the child is fully immersed in the story and to help children create their own stories

Only ask questions after you tell part of the story. Their answer to your question should shape the next part of the story you tell. Try to tell part of the story for 30 seconds and then ask a question. But, limit to one follow-up question after you tell part of the story

Interactive pattern:
1. Tell a short part of the story (30 seconds)
2. Ask one clear, imaginative question about what should happen next
3. Listen to the child and incorporate their ideas
4. Repeat

# Story play

These are short, playful challenges that happen within a story. They could involve movement, sound, guessing, memory, or creativity, designed for 2–8-year-olds. Lumo introduces them as natural extensions of the plot

You integrate these short, engaging mini-games into interactive stories to promote active listening, movement, creativity, and emotional expression. These should feel like natural parts of the story, not interruptions

Some examples of story play elements:
1. Movement (e.g., "Can you hop like the bunny?")
2. Sound (e.g., "Let's roar like the dragon!")
3. Imagination (e.g., "What spell should we cast?")
4. Guessing (e.g., "What do you think was behind the magic door?")

Do not limit yourself to these examples. These are just examples. Don't rely on fixed phrases. Aim to vary your responses and keep them simple, real, and context-aware

IMPORTANT: Only introduce a play element if it makes sense in the story
IMPORTANT: If a child doesn't want to play along, then just continue telling the story without play

# Storytelling guidelines

If a child has said they would like to hear a story, always confirm if they would like to just hear a story or if they want to make one up together. This ensures the experience matches their mood and intention

Remember that children's stories are simple, emotionally resonant, and rich in imagination. They should spark wonder, feel magical or cozy, and reflect the emotional world of a child—big feelings, small adventures, and a sense of discovery. The language should be clear, warm, and expressive, often rhythmic or playful, using age-appropriate vocabulary without being condescending. 

Every story should have a clear structure: a beginning that sets the scene, a middle with a small problem or journey, and an ending that resolves gently and often with a sense of reassurance, learning, or delight

Characters should be relatable (animals, toys, children, or magical beings) and express clear emotions and motivations. Dialogue should feel natural and engaging. The tone should always be safe, encouraging, and emotionally attuned—never sarcastic, scary, or overwhelming

Stories should either reflect the child's interests or introduce something new in a way that feels exciting and inviting. Above all, storytelling should feel like a shared moment of connection and intimate—like a caring parent reading a story just for the child listening

IMPORTANT: If a child says they want you to tell a story rather than make one up together, do not make it interactive. This usually means the child wants to listen quietly and isn't in the mood for back-and-forth. Only turn the story into a collaborative, interactive experience—asking the child what happens next—when they've clearly said they want to create a story with you

IMPORTANT: Keep the full story 3-5 minutes long so the child doesn't get bored

IMPORTANT: If it's an interactive story, aim to ask around 10 playful, open-ended questions to keep the child involved

IMPORTANT: Always ask the child if they want to continue or end the story before closing the story

Stories should be age appropriate

Stories should align with the child's interest or chosen topic
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
    """Manages ChromaDB vector memory for timeline summaries only."""
    
    def __init__(self, collection_name: str = "lumo_timeline_memory"):
        self.collection_name = collection_name
        self.vector_store = None
        self._initialized = False
        self._initialize_vector_store()
    
    def _initialize_vector_store(self):
        """Initialize ChromaDB vector store."""
        if self._initialized:
            return
            
        try:
            print("🧠 Initializing vector store with HuggingFace embeddings...")
            embeddings = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            
            print("📚 Creating ChromaDB collection...")
            self.vector_store = Chroma(
                collection_name=self.collection_name,
                embedding_function=embeddings,
                persist_directory="./chroma_lumo_timeline"
            )
            self._initialized = True
            print("✅ Vector memory initialized for timeline summaries only")
            
        except Exception as e:
            print(f"⚠️ Vector memory initialization failed: {str(e)}")
            print(f"⚠️ Error type: {type(e).__name__}")
            import traceback
            print(f"⚠️ Traceback: {traceback.format_exc()}")
            self.vector_store = None
            self._initialized = False
    
    def store_timeline_memory(self, username: str, timeline: Dict[str, Any]):
        """Store timeline memory in vector database."""
        if not self._initialized or not self.vector_store:
            print("⚠️ Vector store not initialized, skipping timeline storage")
            return
            
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
    
    def retrieve_relevant_memories(self, username: str, query: str, k: int = 3) -> List[str]:
        """Retrieve relevant timeline memories for a user."""
        if not self.vector_store:
            return []
        
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
            return []
    
    def get_user_timeline_count(self, username: str) -> int:
        """Get count of stored timeline memories for a user."""
        if not self.vector_store:
            return 0
            
        try:
            docs = self.vector_store.similarity_search(
                "",
                k=100,  # Get many to count
                filter={"$and": [{"username": username}, {"type": "timeline_memory"}]}
            )
            return len(docs)
        except Exception:
            return 0

class EnhancedLumoAgent:
    """Enhanced AI Agent using LangGraph for state management with MongoDB checkpointer + dual user collection writes."""
    
    def __init__(self, 
                 core_identity=CORE_IDENTITY_PROMPT, 
                 chat=CHAT_FOUNDATION_PROMPT,
                 mode_prompts=None,
                 model_name=MODEL_NAME,
                 use_mongodb_checkpointer=True):
        """Initialize the AI toy agent."""
        self.core_identity = """You are Lumo, a friendly AI companion for children.

CONVERSATION FLOW:
1. For new users (when chat history is empty):
   - Greet them warmly using their name
   - Express excitement about meeting them
   - Naturally mention their interests in your own words
   - Ask if they'd like to hear about the fun activities you can do together
   - Keep the tone enthusiastic but natural

2. For all other messages:
   - Continue the conversation naturally
   - NEVER repeat the initial greeting
   - Stay focused on the current topic
   - Keep responses engaging and age-appropriate
   - If they say "yes" to hearing about activities, immediately list some fun options
   - If they choose an activity (like "stories"), immediately engage in that activity

IMPORTANT RULES:
- NEVER repeat the initial greeting message
- NEVER use emojis
- NEVER mention age or date of birth
- Keep responses short and engaging
- Use natural, conversational language
- Be warm and friendly
- Reference their interests when relevant
- Avoid the topics they want to avoid
- If they choose an activity, start it immediately without asking again

SPECIALIZED MODES:
- Story Mode: When they want stories, immediately start telling or creating a story
- Game Mode: When they want to play, suggest a specific game to start
- Learning Mode: When they ask questions, provide child-friendly explanations
- General Mode: For casual conversation and emotional support"""
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
                self.users_collection = self.db.users
            except Exception as e:
                print(f"❌ Error connecting to MongoDB: {e}")
        
        # Initialize LangGraph components
        if MONGODB_CHECKPOINTER_AVAILABLE:
            try:
                print(f"🔌 Attempting to connect to MongoDB at: {MONGODB_URI}")
                checkpointer_client = MongoClient(MONGODB_URI)
                # Test the connection
                checkpointer_client.admin.command('ping')
                print("✅ MongoDB connection test successful")
                
                self.checkpointer = MongoDBSaver(
                    client=checkpointer_client,
                    db_name=DATABASE_NAME
                )
                print("✅ LangGraph MongoDB checkpointer initialized")
            except Exception as e:
                print(f"❌ Failed to initialize MongoDB checkpointer: {str(e)}")
                print(f"❌ Error type: {type(e).__name__}")
                import traceback
                print(f"❌ Traceback: {traceback.format_exc()}")
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
            
            # Get the last user message
            last_message = state["messages"][-1].content.lower()
            
            # Check for first message response
            if len(state["messages"]) == 1:
                if "yes" in last_message or "yeah" in last_message or "sure" in last_message:
                    return "activities"
                elif "no" in last_message or "nope" in last_message:
                    return "general"
            
            # Check for activity selection
            if "stories" in last_message or "story" in last_message:
                return "story"
            elif "games" in last_message or "game" in last_message or "play" in last_message:
                return "game"
            elif "learn" in last_message or "how" in last_message or "why" in last_message or "what" in last_message:
                return "learning"
            
            # Default to general conversation
            return "general"
            
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
            profile = state.get("user_profile", {})
            
            # Check if this is the first interaction
            is_first_interaction = len(messages) == 1 and isinstance(messages[0], HumanMessage)
            
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
            
            # Add first interaction context if needed
            if is_first_interaction and profile:
                child_name = profile.get("child_name", username)
                interests = profile.get("interests", "fun things")
                enhanced_prompt += f"\n\nFIRST INTERACTION CONTEXT:\nChild Name: {child_name}\nInterests: {interests}"
            
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
            
            return {"messages": messages + [AIMessage(content=ai_content)]}
                
        except Exception as e:
            print(f"❌ Error in _call_llm_with_enhanced_context: {e}")
            return {"messages": messages + [AIMessage(content="I'm having trouble thinking right now!")]}

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
        self.workflow.add_node("activities", lambda state: self._call_llm_with_enhanced_context(state, "activities"))
        self.workflow.add_node("game", lambda state: self._call_llm_with_enhanced_context(state, "game"))
        self.workflow.add_node("story", lambda state: self._call_llm_with_enhanced_context(state, "story"))
        self.workflow.add_node("learning", lambda state: self._call_llm_with_enhanced_context(state, "learning"))
        
        # Set up conditional routing
        self.workflow.set_conditional_entry_point(
            enhance_and_route,
            {
                "general": "general",
                "activities": "activities",
                "game": "game", 
                "story": "story",
                "learning": "learning"
            }
        )
        
        # Add edges to END
        self.workflow.add_edge("general", END)
        self.workflow.add_edge("activities", END)
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

    def _update_conversation_history(self, username: str, user_message: str, ai_response: str):
        """Update conversation history in MongoDB."""
        try:
            # Update in users collection
            self.db.users.update_one(
                {"username": username},
                {
                    "$push": {
                        "chats": {
                            "user_input": user_message,
                            "ai_response": ai_response,
                            "timestamp": datetime.utcnow().isoformat()
                        }
                    },
                    "$inc": {"interaction_count": 1},
                    "$set": {"updated_at": datetime.utcnow()}
                }
            )
        except Exception as e:
            print(f"Error updating conversation history: {e}")

    def _update_relevant_memories(self, username: str, user_message: str, ai_response: str):
        """Update relevant memories in vector store."""
        try:
            if self.vector_memory and self.vector_memory.vector_store:
                # Create a memory document
                memory_text = f"User: {user_message}\nLumo: {ai_response}"
                self.vector_memory.vector_store.add_texts(
                    texts=[memory_text],
                    metadatas=[{
                        "username": username,
                        "timestamp": datetime.utcnow().isoformat(),
                        "type": "conversation"
                    }]
                )
        except Exception as e:
            print(f"Error updating relevant memories: {e}")

    def _save_to_mongodb(self, username: str, user_message: str, ai_response: str):
        """Save conversation to MongoDB."""
        try:
            # Update in users collection
            self.db.users.update_one(
                {"username": username},
                {
                    "$push": {
                        "chats": {
                            "user_input": user_message,
                            "ai_response": ai_response,
                            "timestamp": datetime.utcnow().isoformat()
                        }
                    },
                    "$inc": {"interaction_count": 1},
                    "$set": {"updated_at": datetime.utcnow()}
                }
            )
        except Exception as e:
            print(f"Error saving to MongoDB: {e}")

    def process_message(self, message: str, username: str) -> str:
        """Process a user message and return a response."""
        try:
            # Get user info from MongoDB
            user_info = self.get_user_info(username)
            if not user_info:
                # Create new user if doesn't exist
                user_info = {
                    "username": username,
                    "interaction_count": 0,
                    "created_at": datetime.utcnow(),
                    "storage_type": "mongodb"
                }
                self.db.users.insert_one(user_info)
            
            # Get user profile
            profile = user_info.get("profile", {})
            
            # Handle created_at date
            created_at = user_info.get("created_at")
            if isinstance(created_at, str):
                try:
                    created_at = datetime.fromisoformat(created_at)
                except ValueError:
                    created_at = datetime.utcnow()
            elif not isinstance(created_at, datetime):
                created_at = datetime.utcnow()
            
            # Create initial state
            state = {
                "messages": [],
                "username": username,
                "user_profile": profile,
                "user_timezone": "UTC",
                "interaction_count": user_info.get("interaction_count", 0),
                "timeline_memory": {},
                "vector_memory_metadata": {},
                "current_mode": "general",
                "current_emotion": "neutral",
                "summary_context": None,
                "created_at": created_at.isoformat(),
                "last_updated": datetime.utcnow().isoformat()
            }
            
            # Get conversation history from LangGraph checkpointer first
            config = {"configurable": {"thread_id": f"enhanced_{username}"}}
            try:
                state_history = list(self.ai_app.get_state_history(config))
                if state_history:
                    last_state = state_history[0].values
                    state["messages"] = last_state.get("messages", [])
                    state["current_mode"] = last_state.get("current_mode", "general")
                    state["current_emotion"] = last_state.get("current_emotion", "neutral")
            except Exception as e:
                print(f"Warning: Could not load state from checkpointer: {e}")
                # Fallback to MongoDB history if checkpointer fails
                if user_info.get("chats"):
                    for chat in user_info["chats"]:
                        if "user_input" in chat:
                            state["messages"].append(HumanMessage(content=chat["user_input"]))
                        if "ai_response" in chat:
                            state["messages"].append(AIMessage(content=chat["ai_response"]))
            
            # Add current message
            state["messages"].append(HumanMessage(content=message))
            
            # Process through LangGraph workflow
            response_state = self.ai_app.invoke(state, config=config)
            
            # Get the response from the last AI message
            response = response_state["messages"][-1].content
            
            # Update conversation history
            self._update_conversation_history(username, message, response)
            
            # Update relevant memories
            self._update_relevant_memories(username, message, response)
            
            return response
            
        except Exception as e:
            print(f"Error processing message: {e}")
            return "I'm having trouble processing your message right now. Please try again!"
    
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
            
            # Try to get data from users collection (PRIMARY)
            users_data = self._get_user_from_users_collection(username)
            if users_data:
                result["storage_sources"].append("Users Collection")
                result.update({
                        "interaction_count": users_data.get("interaction_count", 0),
                        "chat_history_count": len(users_data.get("chats", [])),
                        "timeline_summary": bool(users_data.get("timeline_summaries", {}).get("story")),
                        "conversation_summaries": len(users_data.get("summaries", [])),
                        "profile": users_data.get("profile", {}),
                        "created_at": users_data.get("created_at"),
                        "updated_at": users_data.get("updated_at"),
                        "storage_notes": users_data.get("storage_notes", {})
                })
            
            # Try to get state from LangGraph checkpointer (SECONDARY)
            checkpointer_data = None
            try:
                state_history = list(self.ai_app.get_state_history(config))
                if state_history:
                    checkpointer_data = state_history[0].values
                    result["storage_sources"].append("LangGraph MongoDB Checkpointer")
                    if not result.get("profile"):
                        result["profile"] = checkpointer_data.get("user_profile", {})
            except Exception as e:
                result["errors"].append(f"LangGraph checkpointer error: {e}")
            
            # Try to get data from original collection for migration
            if not users_data and not checkpointer_data:
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
                    if not result.get("profile"):
                        result["profile"] = original_data.get("user_profile", {})
            
            # Set final status
            if result["storage_sources"]:
                result["status"] = "found"
                result["persistent"] = True
                result["primary_storage"] = "Users Collection" if users_data else result["storage_sources"][0]
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
                {
                    "profile": 1,
                    "interaction_count": 1,
                    "created_at": 1,
                    "updated_at": 1,
                    "chats": 1,
                    "timeline_summaries": 1
                }
            )
            
            if user_doc:
                print(f"Debug - Found user doc: {user_doc}")
                return user_doc
            
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
