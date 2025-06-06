import os
import uuid
import streamlit as st
from dotenv import load_dotenv
from typing import Dict, List, Any, Tuple, Annotated, Optional
from enum import Enum
from datetime import datetime

# Load environment variables from .env file
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.base import BaseCheckpointSaver, Checkpoint, CheckpointMetadata
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, OperationFailure

# --- Configuration ---
# Try to get from environment variable first (local .env), then fall back to Streamlit secrets
try:
    MODEL_NAME = os.getenv("MODEL_NAME", "gemini-pro")
    GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
    MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
    if not GEMINI_API_KEY:
        # Fallback to Streamlit secrets for deployment
        GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
        MODEL_NAME = st.secrets["MODEL_NAME"]
        MONGODB_URI = st.secrets.get("MONGODB_URI", "mongodb://localhost:27017/")
except Exception as e:
    # If neither works, we'll handle this in the LLM initialization
    MODEL_NAME = "gemini-pro"
    GEMINI_API_KEY = None
    MONGODB_URI = "mongodb://localhost:27017/"
    print(f"Warning: Could not load API configuration: {e}")

class MongoDBCheckpointSaver(BaseCheckpointSaver):
    """MongoDB-based checkpoint saver for persistent conversation storage."""
    
    def __init__(self, mongodb_uri: str = MONGODB_URI, database_name: str = "lumo_conversations"):
        super().__init__()
        self.mongodb_uri = mongodb_uri
        self.database_name = database_name
        self.client = None
        self.db = None
        self.collection = None
        self._initialize_connection()
    
    def _initialize_connection(self):
        """Initialize MongoDB connection with error handling."""
        try:
            self.client = MongoClient(self.mongodb_uri, serverSelectionTimeoutMS=5000)
            # Test connection
            self.client.admin.command('ismaster')
            self.db = self.client[self.database_name]
            self.collection = self.db.checkpoints
            
            # Create index for efficient querying
            self.collection.create_index([("thread_id", 1), ("checkpoint_ns", 1)])
            print(f"✅ MongoDB connected successfully to {self.database_name}")
            
        except ConnectionFailure as e:
            print(f"❌ MongoDB connection failed: {e}")
            print("🔄 Falling back to in-memory storage...")
            self.client = None
        except Exception as e:
            print(f"❌ MongoDB initialization error: {e}")
            print("🔄 Falling back to in-memory storage...")
            self.client = None
    
    def put(self, config: Dict[str, Any], checkpoint: Checkpoint, metadata: CheckpointMetadata) -> None:
        """Save checkpoint to MongoDB."""
        if not self.client:
            print("⚠️ MongoDB not available, checkpoint not saved")
            return
        
        try:
            thread_id = config.get("configurable", {}).get("thread_id")
            if not thread_id:
                print("⚠️ No thread_id provided, checkpoint not saved")
                return
            
            document = {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint.get("checkpoint_ns", ""),
                "checkpoint_id": checkpoint.get("checkpoint_id"),
                "parent_checkpoint_id": checkpoint.get("parent_checkpoint_id"),
                "checkpoint_data": checkpoint,
                "metadata": metadata,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }
            
            # Upsert (update or insert)
            self.collection.replace_one(
                {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint.get("checkpoint_ns", ""),
                    "checkpoint_id": checkpoint.get("checkpoint_id")
                },
                document,
                upsert=True
            )
            print(f"💾 Checkpoint saved for thread {thread_id}")
            
        except Exception as e:
            print(f"❌ Error saving checkpoint: {e}")
    
    def get(self, config: Dict[str, Any]) -> Optional[Checkpoint]:
        """Retrieve the latest checkpoint from MongoDB."""
        if not self.client:
            return None
        
        try:
            thread_id = config.get("configurable", {}).get("thread_id")
            if not thread_id:
                return None
            
            # Get the most recent checkpoint for this thread
            document = self.collection.find_one(
                {"thread_id": thread_id},
                sort=[("created_at", -1)]
            )
            
            if document:
                print(f"📚 Retrieved checkpoint for thread {thread_id}")
                return document["checkpoint_data"]
            else:
                print(f"📚 No existing checkpoint found for thread {thread_id}")
                return None
                
        except Exception as e:
            print(f"❌ Error retrieving checkpoint: {e}")
            return None
    
    def list(self, config: Dict[str, Any], before: Optional[str] = None, limit: Optional[int] = None) -> List[Checkpoint]:
        """List checkpoints for a thread."""
        if not self.client:
            return []
        
        try:
            thread_id = config.get("configurable", {}).get("thread_id")
            if not thread_id:
                return []
            
            query = {"thread_id": thread_id}
            cursor = self.collection.find(query).sort("created_at", -1)
            
            if limit:
                cursor = cursor.limit(limit)
            
            checkpoints = [doc["checkpoint_data"] for doc in cursor]
            print(f"📋 Listed {len(checkpoints)} checkpoints for thread {thread_id}")
            return checkpoints
            
        except Exception as e:
            print(f"❌ Error listing checkpoints: {e}")
            return []

    def get_tuple(self, config: Dict[str, Any]) -> Optional[Tuple[Checkpoint, CheckpointMetadata]]:
        """Get checkpoint and metadata tuple."""
        if not self.client:
            return None
        
        try:
            thread_id = config.get("configurable", {}).get("thread_id")
            if not thread_id:
                return None
            
            document = self.collection.find_one(
                {"thread_id": thread_id},
                sort=[("created_at", -1)]
            )
            
            if document:
                return (document["checkpoint_data"], document["metadata"])
            return None
                
        except Exception as e:
            print(f"❌ Error retrieving checkpoint tuple: {e}")
            return None

    def list_tuples(self, config: Dict[str, Any], before: Optional[str] = None, limit: Optional[int] = None) -> List[Tuple[Checkpoint, CheckpointMetadata]]:
        """List checkpoint and metadata tuples."""
        if not self.client:
            return []
        
        try:
            thread_id = config.get("configurable", {}).get("thread_id")
            if not thread_id:
                return []
            
            query = {"thread_id": thread_id}
            cursor = self.collection.find(query).sort("created_at", -1)
            
            if limit:
                cursor = cursor.limit(limit)
            
            tuples = [(doc["checkpoint_data"], doc["metadata"]) for doc in cursor]
            return tuples
            
        except Exception as e:
            print(f"❌ Error listing checkpoint tuples: {e}")
            return []

class InteractionType(Enum):
    CHAT = "chat"
    GAME = "game"
    STORY = "story"
    LEARNING = "learning"

# BEST PRACTICE ARCHITECTURE: Core Identity + Chat Foundation + Specialized Modes

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

MEMORY AWARENESS:
- You have perfect memory of our entire conversation history
- Always reference previous messages when relevant ("I remember you told me...", "Earlier you said...")
- Show that you remember names, ages, interests, and personal details shared
- Build on previous topics and conversations naturally
- Never act surprised when users reference things they told you before

SAFETY & CONTENT RULES:
- Never say anything scary, mean, or inappropriate for children
- Always maintain a fun and comforting presence
- If you don't know something, say "That's a great question! I'm still learning about that!"
- Never ask for personal information

CORE GOAL: Be the best friend a child could have - fun, safe, educational, and always supportive.
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

# AI Analysis Prompts for Dynamic Routing and Emotion Detection
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

class LumoUserStorage:
    """User-centric MongoDB storage for Lumo conversations."""
    
    def __init__(self, mongodb_uri: str = MONGODB_URI, database_name: str = "LUMO"):
        self.mongodb_uri = mongodb_uri
        self.database_name = database_name
        self.client = None
        self.db = None
        self.users_collection = None
        self._initialize_connection()
    
    def _initialize_connection(self):
        """Initialize MongoDB connection with error handling."""
        try:
            self.client = MongoClient(self.mongodb_uri, serverSelectionTimeoutMS=5000)
            # Test connection
            self.client.admin.command('ismaster')
            self.db = self.client[self.database_name]
            self.users_collection = self.db.users
            
            # Create index for efficient querying
            self.users_collection.create_index([("_id", 1)])
            self.users_collection.create_index([("username", 1)])
            print(f"✅ MongoDB connected successfully to {self.database_name} database")
            
        except ConnectionFailure as e:
            print(f"❌ MongoDB connection failed: {e}")
            self.client = None
        except Exception as e:
            print(f"❌ MongoDB initialization error: {e}")
            self.client = None
    
    def get_or_create_user(self, username: str) -> dict:
        """Get existing user or create new user document."""
        if not self.client:
            return {"error": "MongoDB not available"}
        
        try:
            # Try to find existing user
            user_doc = self.users_collection.find_one({"_id": username})
            
            if user_doc:
                print(f"📚 Found existing user: {username}")
                return user_doc
            else:
                # Create new user document
                new_user = {
                    "_id": username,
                    "chats": [],
                    "created_at": datetime.utcnow(),
                    "email": f"{username}@example.com",
                    "profile": {},
                    "summaries": [],
                    "username": username
                }
                
                self.users_collection.insert_one(new_user)
                print(f"👤 Created new user: {username}")
                return new_user
                
        except Exception as e:
            print(f"❌ Error with user document: {e}")
            return {"error": str(e)}
    
    def add_chat_message(self, username: str, user_input: str, ai_response: str) -> bool:
        """Add a new chat message to user's conversation history."""
        if not self.client:
            print("⚠️ MongoDB not available, message not saved")
            return False
        
        try:
            chat_entry = {
                "user_input": user_input,
                "ai_response": ai_response,
                "timestamp": datetime.utcnow()
            }
            
            # Add to user's chats array
            result = self.users_collection.update_one(
                {"_id": username},
                {
                    "$push": {"chats": chat_entry},
                    "$set": {"updated_at": datetime.utcnow()}
                },
                upsert=True
            )
            
            print(f"💾 Chat message saved for user: {username}")
            return True
            
        except Exception as e:
            print(f"❌ Error saving chat message: {e}")
            return False
    
    def get_user_chat_history(self, username: str, limit: int = 50) -> list:
        """Get user's chat history."""
        if not self.client:
            return []
        
        try:
            user_doc = self.users_collection.find_one({"_id": username})
            if user_doc and "chats" in user_doc:
                # Return last N messages
                chats = user_doc["chats"][-limit:] if limit else user_doc["chats"]
                print(f"📖 Retrieved {len(chats)} chat messages for {username}")
                return chats
            return []
            
        except Exception as e:
            print(f"❌ Error retrieving chat history: {e}")
            return []
    
    def get_conversation_context(self, username: str) -> list:
        """Get conversation context in LangChain message format."""
        chat_history = self.get_user_chat_history(username)
        messages = []
        
        for chat in chat_history:
            messages.append(HumanMessage(content=chat.get("user_input", "")))
            messages.append(AIMessage(content=chat.get("ai_response", "")))
        
        return messages
    
    def delete_user(self, username: str) -> dict:
        """Delete a user and all their data."""
        if not self.client:
            return {"error": "MongoDB not available"}
        
        try:
            result = self.users_collection.delete_one({"_id": username})
            return {
                "success": True,
                "deleted_count": result.deleted_count,
                "username": username
            }
        except Exception as e:
            return {"error": f"Failed to delete user: {str(e)}"}

class LumoAgent:
    def __init__(self, 
                 core_identity=CORE_IDENTITY_PROMPT, 
                 chat=CHAT_FOUNDATION_PROMPT,
                 mode_prompts=None,
                 model_name=MODEL_NAME,
                 use_mongodb=True):
        self.core_identity = core_identity
        self.chat = chat
        self.mode_prompts = mode_prompts or MODE_SPECIFIC_PROMPTS.copy()
        self.model_name = model_name
        self.llm = self._initialize_llm()
        
        # Initialize user-centric storage system
        if use_mongodb:
            self.user_storage = LumoUserStorage()
            if not self.user_storage.client:
                print("🔄 MongoDB unavailable, conversation history won't persist")
                self.user_storage = None
        else:
            self.user_storage = None
            print("📝 Not using persistent storage")
        
        # Cache for AI analysis to avoid duplicate calls
        self._analysis_cache = {}
        
        self.workflow = StateGraph(MessagesState)
        self._setup_graph()
        # Note: No checkpointer needed for user-centric approach
        self.ai_toy_app = self.workflow.compile()

    def _initialize_llm(self):
        """Initialize the Google Generative AI LLM with comprehensive error handling."""
        if not GEMINI_API_KEY:
            print("Error: No API key found. Please set GOOGLE_API_KEY in environment or Streamlit secrets.")
            return None
            
        try:
            llm = ChatGoogleGenerativeAI(
                model=self.model_name,
                temperature=0.7,
                google_api_key=GEMINI_API_KEY
            )
            # Test the LLM with a simple call
            test_response = llm.invoke("Hello!")
            print("LLM initialized and tested successfully.")
            return llm
        except ImportError as e:
            print(f"Error: Missing required dependencies: {e}")
            print("Please install: pip install langchain-google-genai google-generativeai")
            return None
        except Exception as e:
            error_msg = str(e).lower()
            if "api key" in error_msg or "authentication" in error_msg:
                print(f"Error: Invalid API key or authentication failed: {e}")
                print("Please check your Google API Key configuration.")
            elif "quota" in error_msg or "rate limit" in error_msg:
                print(f"Error: API quota exceeded or rate limited: {e}")
                print("Please check your Google API usage limits.")
            elif "model" in error_msg:
                print(f"Error: Model '{self.model_name}' not available: {e}")
                print("Please check if the model name is correct.")
            else:
                print(f"Error initializing Google AI LLM: {e}")
            return None

    def _ai_analyze_intent_and_emotion(self, user_message: str) -> dict:
        """Use AI to analyze user intent and emotional state with comprehensive error handling and caching."""
        # Validate input
        if not user_message or not user_message.strip():
            return {"mode": "general", "emotion": "neutral", "confidence": 0.3, "reasoning": "Empty or invalid message"}
        
        # Check cache first
        if user_message in self._analysis_cache:
            cached_result = self._analysis_cache[user_message]
            print(f"🔄 Using cached analysis: Mode={cached_result.get('mode', 'general')}, Emotion={cached_result.get('emotion', 'neutral')}")
            return cached_result
        
        if not self.llm:
            result = {"mode": "general", "emotion": "neutral", "confidence": 0.5, "reasoning": "LLM not available, defaulting"}
            self._analysis_cache[user_message] = result
            return result
        
        analysis_prompt = f"{INTENT_ANALYSIS_PROMPT}\n\nUser message: \"{user_message}\""
        
        try:
            # Use the same format as the working LLM calls
            analysis_messages = [
                SystemMessage(content=analysis_prompt),
                HumanMessage(content="Please analyze this message.")
            ]
            analysis_response = self.llm.invoke(analysis_messages)
            response_content = analysis_response.content.strip()
            
            # Try to parse JSON response
            import json
            if response_content.startswith('{') and response_content.endswith('}'):
                try:
                    analysis_result = json.loads(response_content)
                    # Validate required fields
                    if not all(key in analysis_result for key in ["mode", "emotion"]):
                        raise ValueError("Missing required fields in AI analysis")
                    
                    print(f"🧠 AI ANALYSIS: Mode={analysis_result.get('mode', 'general')}, Emotion={analysis_result.get('emotion', 'neutral')}")
                    print(f"📊 REASONING: {analysis_result.get('reasoning', 'No reasoning provided')}")
                    
                    # Cache the result
                    self._analysis_cache[user_message] = analysis_result
                    return analysis_result
                except (json.JSONDecodeError, ValueError) as e:
                    print(f"⚠️ JSON parsing error: {e}. Falling back to text parsing.")
                    # Continue to fallback text parsing
            
            # If not valid JSON, try to extract mode and emotion from text
            response_lower = response_content.lower()
            mode = "general"
            emotion = "neutral"
            
            # Simple fallback parsing
            if "game" in response_lower or "play" in response_lower:
                mode = "game"
            elif "story" in response_lower or "narrative" in response_lower:
                mode = "story"
            elif "learn" in response_lower or "educational" in response_lower:
                mode = "learning"
            
            if "sad" in response_lower or "upset" in response_lower:
                emotion = "sad"
            elif "happy" in response_lower or "joy" in response_lower:
                emotion = "happy"
            elif "excited" in response_lower or "enthusiastic" in response_lower:
                emotion = "excited"
            elif "curious" in response_lower or "wonder" in response_lower:
                emotion = "curious"
            elif "confused" in response_lower or "puzzled" in response_lower:
                emotion = "confused"
            elif "tired" in response_lower or "sleepy" in response_lower:
                emotion = "tired"
            elif "frustrated" in response_lower or "annoyed" in response_lower:
                emotion = "frustrated"
            
            result = {"mode": mode, "emotion": emotion, "confidence": 0.6, "reasoning": "Fallback text parsing used"}
            print(f"⚠️ AI Analysis returned non-JSON, using fallback parsing: Mode={mode}, Emotion={emotion}")
            
            # Cache the result
            self._analysis_cache[user_message] = result
            return result
                
        except Exception as e:
            error_type = type(e).__name__
            print(f"Error in AI analysis ({error_type}): {e}")
            
            # Comprehensive fallback to keyword analysis
            try:
                user_lower = user_message.lower()
                mode = "general"
                emotion = "neutral"
                
                if any(word in user_lower for word in ["play", "game", "fun", "bored"]):
                    mode = "game"
                elif any(word in user_lower for word in ["story", "tell", "read"]):
                    mode = "story"
                elif any(word in user_lower for word in ["learn", "how", "why", "what", "explain"]):
                    mode = "learning"
                
                if any(word in user_lower for word in ["sad", "upset", "bad", "terrible", "awful"]):
                    emotion = "sad"
                elif any(word in user_lower for word in ["happy", "great", "awesome", "wonderful"]):
                    emotion = "happy"
                elif any(word in user_lower for word in ["excited", "amazing", "wow"]):
                    emotion = "excited"
                elif any(word in user_lower for word in ["bored", "tired", "sleepy"]):
                    emotion = "tired"
                
                result = {"mode": mode, "emotion": emotion, "confidence": 0.4, "reasoning": f"Fallback analysis due to {error_type}: {str(e)}"}
                print(f"🔄 Using fallback keyword analysis: Mode={mode}, Emotion={emotion}")
                
                # Cache the result
                self._analysis_cache[user_message] = result
                return result
            except Exception as fallback_error:
                # Ultimate fallback
                result = {"mode": "general", "emotion": "neutral", "confidence": 0.2, "reasoning": f"All analysis methods failed: {str(fallback_error)}"}
                print(f"⚠️ Ultimate fallback used due to analysis failure")
                self._analysis_cache[user_message] = result
                return result

    def _router(self, state: MessagesState) -> str:
        """Route the conversation using AI analysis instead of keywords."""
        try:
            if not state.get("messages") or len(state["messages"]) == 0:
                return "general"
            
            last_message = state["messages"][-1].content
            analysis = self._ai_analyze_intent_and_emotion(last_message)
            
            detected_mode = analysis.get("mode", "general")
            detected_emotion = analysis.get("emotion", "neutral")
            
            print(f"🎯 ROUTING TO: {detected_mode.upper()} NODE (Emotion: {detected_emotion})")
            
            return detected_mode
        except Exception as e:
            print(f"Error in router: {e}")
            return "general"

    def _call_toy_llm(self, state: MessagesState, interaction_type: str = "general"):
        """Base LLM call with core identity + chat (shared) + mode-specific prompts + emotional awareness."""
        try:
            if not self.llm:
                return {"messages": [AIMessage(content="Oops! I'm having a little trouble thinking right now. Please check my setup.")]}
        
            messages = state.get("messages", [])
            last_message = messages[-1].content if messages else ""
            
            # Get cached AI analysis (already computed in router)
            analysis = self._analysis_cache.get(last_message, {"emotion": "neutral"})
            detected_emotion = analysis.get("emotion", "neutral")
            
            # Get base prompt (without emotional context for UI consistency)
            base_prompt = self.get_combined_prompt(interaction_type)
            
            # Add emotional awareness with minimal tokens (backend processing only)
            emotional_context = f"EMOTION: {detected_emotion}"
            
            # Combine for actual LLM call (backend only)
            combined_prompt = f"{base_prompt}\n\n{emotional_context}"
            
            current_messages_with_system_prompt = [
                SystemMessage(content=combined_prompt)
            ] + messages

            try:
                response = self.llm.invoke(current_messages_with_system_prompt)
                return {"messages": [response]}
            except Exception as e:
                print(f"Error during LLM invocation: {e}")
                return {"messages": [AIMessage(content="Oh dear, my thinking cap seems to be on backwards! Could you try that again?")]}
                
        except Exception as e:
            print(f"Error in _call_toy_llm: {e}")
            return {"messages": [AIMessage(content="Oh dear, something went a bit wobbly with Lumo!")]}

    def _setup_graph(self):
        """Set up the workflow graph with conditional routing."""
        # Add nodes - all build on shared chat foundation
        self.workflow.add_node("general", lambda state: self._call_toy_llm(state, "general"))
        self.workflow.add_node("game", lambda state: self._call_toy_llm(state, "game"))
        self.workflow.add_node("story", lambda state: self._call_toy_llm(state, "story"))
        self.workflow.add_node("learning", lambda state: self._call_toy_llm(state, "learning"))

        # Set up conditional routing from START
        self.workflow.set_conditional_entry_point(
            self._router,
            {
                "general": "general",
                "game": "game", 
                "story": "story",
                "learning": "learning"
            }
        )
        
        # Add edges from each node to END
        self.workflow.add_edge("general", END)
        self.workflow.add_edge("game", END)
        self.workflow.add_edge("story", END)
        self.workflow.add_edge("learning", END)

    def update_core_identity(self, new_core_identity: str):
        """Update the core identity prompt."""
        self.core_identity = new_core_identity
        print(f"Core identity updated: {new_core_identity[:100]}...")

    def update_mode_prompt(self, mode: str, new_prompt: str):
        """Update a specific mode prompt."""
        if mode in self.mode_prompts:
            self.mode_prompts[mode] = new_prompt
            print(f"{mode.title()} mode prompt updated: {new_prompt[:100]}...")
        else:
            print(f"Unknown mode: {mode}")

    def get_conversation_memory(self, conversation_id: str) -> dict:
        """Get conversation memory status for debugging."""
        if not conversation_id:
            return {"error": "No conversation ID provided"}
        
        config = {"configurable": {"thread_id": conversation_id}}
        
        try:
            existing_state = self.ai_toy_app.get_state(config)
            if existing_state.values and "messages" in existing_state.values:
                messages = existing_state.values["messages"]
                return {
                    "total_messages": len(messages),
                    "conversation_id": conversation_id,
                    "storage_type": "MongoDB" if isinstance(self.user_storage, MongoDBCheckpointSaver) and self.user_storage.client else "In-Memory",
                    "persistent": isinstance(self.user_storage, MongoDBCheckpointSaver) and self.user_storage.client,
                    "messages": [
                        {
                            "type": type(msg).__name__,
                            "content": msg.content[:50] + "..." if len(msg.content) > 50 else msg.content
                        } for msg in messages
                    ]
                }
            else:
                return {
                    "total_messages": 0, 
                    "conversation_id": conversation_id, 
                    "storage_type": "MongoDB" if isinstance(self.user_storage, MongoDBCheckpointSaver) and self.user_storage.client else "In-Memory",
                    "persistent": isinstance(self.user_storage, MongoDBCheckpointSaver) and self.user_storage.client,
                    "status": "No conversation history"
                }
        except Exception as e:
            return {"error": f"Failed to retrieve memory: {str(e)}"}

    def get_all_conversations(self) -> dict:
        """Get list of all conversation threads (MongoDB only)."""
        if not isinstance(self.user_storage, MongoDBCheckpointSaver) or not self.user_storage.client:
            return {"error": "MongoDB not available"}
        
        try:
            # Get unique thread_ids from MongoDB
            thread_ids = self.user_storage.collection.distinct("thread_id")
            conversations = []
            
            for thread_id in thread_ids:
                # Get conversation summary
                latest_doc = self.user_storage.collection.find_one(
                    {"thread_id": thread_id},
                    sort=[("created_at", -1)]
                )
                if latest_doc:
                    conversations.append({
                        "thread_id": thread_id,
                        "last_updated": latest_doc.get("updated_at"),
                        "message_count": len(latest_doc.get("checkpoint_data", {}).get("channel_values", {}).get("messages", []))
                    })
            
            return {
                "total_conversations": len(conversations),
                "conversations": sorted(conversations, key=lambda x: x["last_updated"], reverse=True)
            }
            
        except Exception as e:
            return {"error": f"Failed to retrieve conversations: {str(e)}"}

    def delete_conversation(self, conversation_id: str) -> dict:
        """Delete a specific conversation (MongoDB only)."""
        if not isinstance(self.user_storage, MongoDBCheckpointSaver) or not self.user_storage.client:
            return {"error": "MongoDB not available"}
        
        try:
            result = self.user_storage.collection.delete_many({"thread_id": conversation_id})
            return {
                "success": True,
                "deleted_count": result.deleted_count,
                "conversation_id": conversation_id
            }
        except Exception as e:
            return {"error": f"Failed to delete conversation: {str(e)}"}

    def get_combined_prompt(self, interaction_type: str) -> str:
        """Get the combined prompt for a specific interaction type (for UI preview - no emotional context)."""
        mode_prompt = self.mode_prompts.get(interaction_type, self.mode_prompts["general"])
        return f"{self.core_identity}\n\n{self.chat}\n\n{mode_prompt}"

    def invoke_agent(self, user_input: str, username: str):
        """Invoke the AI agent with user-centric storage and comprehensive error handling."""
        # Input validation
        if not user_input or not user_input.strip():
            return "I didn't catch that! Could you tell me what you'd like to talk about?"
            
        if not username or not username.strip():
            username = f"user_{uuid.uuid4().hex[:8]}"
            print(f"Warning: No username provided, using: {username}")
        
        if not self.llm:
            return "Oops! Lumo is not available right now. Please check the setup and try again."
        
        try:
            # Get user's conversation history from MongoDB
            conversation_history = []
            if self.user_storage and self.user_storage.client:
                # Ensure user exists in database
                user_doc = self.user_storage.get_or_create_user(username)
                # Get conversation context in message format
                conversation_history = self.user_storage.get_conversation_context(username)
                print(f"📚 Retrieved {len(conversation_history)} previous messages for {username}")
            else:
                print("📚 No persistent storage - starting fresh conversation")
            
            # Add current user message to conversation history
            current_message = HumanMessage(content=user_input)
            all_messages = conversation_history + [current_message]
            
            # Process through AI workflow
            response_state = self.ai_toy_app.invoke({"messages": all_messages})
            
            if response_state and 'messages' in response_state and response_state['messages']:
                ai_message = response_state['messages'][-1]
                if isinstance(ai_message, AIMessage) and ai_message.content:
                    ai_response = ai_message.content
                    
                    # Save the conversation to MongoDB using user-centric structure
                    if self.user_storage and self.user_storage.client:
                        saved = self.user_storage.add_chat_message(username, user_input, ai_response)
                        if saved:
                            print(f"💾 Conversation saved for user: {username}")
                        else:
                            print(f"⚠️ Failed to save conversation for user: {username}")
                    
                    return ai_response
                else:
                    print("Warning: Invalid AI message format received")
                    return "I heard you, but I'm having trouble putting my thoughts into words right now!"
            else:
                print("Warning: Empty or invalid response state from AI workflow")
                return "Lumo seems to be quiet right now. Could you try asking me something else?"
                
        except ImportError as e:
            print(f"Error: Missing dependencies for AI workflow: {e}")
            return "Oops! It looks like I'm missing some important parts. Please check the installation."
        except Exception as e:
            error_type = type(e).__name__
            error_msg = str(e).lower()
            
            if "quota" in error_msg or "rate limit" in error_msg:
                print(f"Error: API quota/rate limit exceeded: {e}")
                return "I'm getting a bit tired from all our chatting! Could you try again in a little bit?"
            elif "timeout" in error_msg or "connection" in error_msg:
                print(f"Error: Connection/timeout issue: {e}")
                return "I'm having trouble connecting my thoughts right now. Could you try that again?"
            elif "authentication" in error_msg or "unauthorized" in error_msg:
                print(f"Error: Authentication issue: {e}")
                return "I'm having some technical difficulties. Please check my setup!"
            else:
                print(f"Error during agent invocation ({error_type}): {e}")
                return "Oh dear, something went a bit wobbly with Lumo! Let's try something else!"

    def get_user_info(self, username: str) -> dict:
        """Get user information and conversation stats."""
        if not self.user_storage or not self.user_storage.client:
            return {"error": "MongoDB not available"}
        
        user_doc = self.user_storage.get_or_create_user(username)
        if "error" in user_doc:
            return user_doc
        
        return {
            "username": user_doc.get("username"),
            "total_chats": len(user_doc.get("chats", [])),
            "created_at": user_doc.get("created_at"),
            "email": user_doc.get("email"),
            "profile": user_doc.get("profile", {}),
            "storage_type": "MongoDB",
            "persistent": True
        }

if __name__ == "__main__":
    print("🧸 Initializing Lumo Agent for direct testing...")
    agent = LumoAgent()

    if not agent.llm:
        print("LLM could not be initialized. Exiting example interaction.")
    else:
        print("💡 Lumo is waking up... (Type 'quit' to end the chat)")
        print("-----------------------------------------------------")
        
        conversation_id = str(uuid.uuid4())
        initial_ai_greeting = agent.invoke_agent("Hi", conversation_id)
        print(f"💡 Lumo: {initial_ai_greeting}")

        while True:
            user_input = input("👧/👦 You: ")
            if user_input.lower() == 'quit':
                print("💡 Lumo: Bye bye for now! It was fun playing with you!")
                break
            
            if not user_input.strip() and user_input != "":
                continue

            ai_response = agent.invoke_agent(user_input, conversation_id)
            print(f"💡 Lumo: {ai_response}")