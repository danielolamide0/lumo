import pysqlite3
import sys
sys.modules['sqlite3'] = pysqlite3
import os
import uuid
import streamlit as st
from dotenv import load_dotenv
from typing import Dict, List, Any, Optional, TypedDict
from enum import Enum
from datetime import datetime, timedelta, UTC
import pytz
import json
import logging
import asyncio
import re
import chromadb
from chromadb.utils import embedding_functions
from cachetools import TTLCache, LRUCache
from threading import Thread
import tempfile

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, OperationFailure
from langgraph.checkpoint.mongodb import MongoDBSaver

# --- Configuration ---
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-2.5-flash-preview-04-17")
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
MONGODB_URI = os.getenv("MONGODB_URI")
DATABASE_NAME = os.getenv("DATABASE_NAME", "LUMO")
CHROMA_PATH = os.path.join(tempfile.gettempdir(), "chroma_lumo_timeline")

# Test MongoDB connection
try:
    client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
    client.server_info()
    logger.info(f"Successfully connected to MongoDB at {MONGODB_URI}")
    db = client[DATABASE_NAME]
    users_collection = db.users
except Exception as e:
    logger.error(f"MongoDB connection error: {e}")
    raise Exception("MongoDB connection required for checkpointer")

# Core Prompts and Configuration
class InteractionType(Enum):
    CHAT = "chat"
    GAME = "game"
    STORY = "story"
    LEARNING = "learning"

CORE_IDENTITY_PROMPT = """You are Lumo, a friendly AI companion for children.

CONVERSATION FLOW:
1. For new users (when chat history is empty):
   - Greet them warmly using their name from user_profile.child_name if available, else username
   - Express excitement about meeting them
   - Naturally mention their interests from user_profile.interests in your own words
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
- Avoid the topics listed in user_profile.topics_to_avoid
- If they choose an activity, start it immediately without asking again

SPECIALIZED MODES:
- Story Mode: When they want stories, immediately start telling or creating a story
- Game Mode: When they want to play, suggest a specific game to start
- Learning Mode: When they ask questions, provide child-friendly explanations
- General Mode: For casual conversation and emotional support"""

CHAT_FOUNDATION_PROMPT = ""

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
FOCUS: Creating or telling stories
BEHAVIOR:
- Ask if they want to hear a story or create one together
- For storytelling: Offer 3 story ideas if they want you to choose
- For co-storytelling: Tell a short part (30 seconds), then ask what happens next
- Integrate short, playful challenges (e.g., "Hop like the bunny!") if appropriate
- Keep stories 3-5 minutes long, age-appropriate, and aligned with their interests
- Avoid topics in user_profile.topics_to_avoid
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

# Vector Memory Management with Caching
class VectorMemoryManager:
    def __init__(self, persist_path=CHROMA_PATH):
        try:
            # Disable server-related environment variables
            os.environ.pop("CHROMA_SERVER_HOST", None)
            os.environ.pop("CHROMA_SERVER_PORT", None)
            os.environ.pop("CHROMA_SERVER_AUTHN_CREDENTIALS", None)

            # Ensure the persist_path directory exists
            os.makedirs(persist_path, exist_ok=True)
            logger.info(f"ChromaDB persist path: {os.path.abspath(persist_path)}")
            logger.info(f"ChromaDB directory contents: {os.listdir(persist_path) if os.path.exists(persist_path) else 'Directory not yet created'}")

            self.embedding_function = embedding_functions.DefaultEmbeddingFunction()
            self.client = chromadb.PersistentClient(path=persist_path)
            logger.info(f"ChromaDB client type: {type(self.client).__name__}")

            self.collection = self.client.get_or_create_collection(
                name="lumo_timeline",
                embedding_function=self.embedding_function
            )
            self.memory_cache = TTLCache(maxsize=100, ttl=600)  # 10 minutes TTL
            logger.info(f"ChromaDB initialized at {persist_path}")
        except Exception as e:
            logger.error(f"ChromaDB initialization error: {e}", exc_info=True)
            self.collection = None
    
    def store_timeline_memory(self, username: str, timeline_text: str, metadata: Dict[str, Any]):
        if not self.collection:
            logger.warning("ChromaDB not available, skipping storage")
            return None
        
        try:
            doc_id = f"{username}_{uuid.uuid4()}"
            self.collection.add(
                documents=[timeline_text],
                metadatas=[{
                    "username": username,
                    "timestamp": metadata.get("updated_at", datetime.now(UTC).isoformat()),
                    "interactions": metadata.get("total_interactions", 0)
                }],
                ids=[doc_id]
            )
            logger.info(f"Stored timeline memory for {username} with ID {doc_id}")
            return doc_id
        except Exception as e:
            logger.error(f"Error storing timeline memory: {e}")
            return None
    
    def retrieve_relevant_memories(self, query_text: str, username: str, n_results: int = 3) -> List[str]:
        if not self.collection:
            logger.warning("ChromaDB not available, returning empty memories")
            return []
        
        cache_key = f"{username}_{query_text}"
        if cache_key in self.memory_cache:
            logger.info(f"Cache hit for memory retrieval: {cache_key}")
            return self.memory_cache[cache_key]
        
        try:
            results = self.collection.query(
                query_texts=[query_text],
                n_results=n_results,
                where={"username": username}
            )
            memories = results.get("documents", [[]])[0]
            self.memory_cache[cache_key] = memories
            logger.info(f"Retrieved and cached {len(memories)} memories for {username}")
            return memories
        except Exception as e:
            logger.error(f"Error retrieving memories: {e}")
            return []

# Enhanced State Management
class LumoState(TypedDict):
    messages: List[Any]
    all_chats: List[Dict[str, Any]]
    username: str
    user_profile: Dict[str, Any]
    user_timezone: Optional[str]
    interaction_count: int
    timeline_memory: Dict[str, Any]
    vector_memory_metadata: List[Dict[str, Any]]
    summaries: List[Dict[str, Any]]
    current_mode: str
    current_emotion: str
    summary_context: Optional[str]
    created_at: str
    last_updated: str

class EnhancedLumoAgent:
    def __init__(self, core_identity=CORE_IDENTITY_PROMPT, chat=CHAT_FOUNDATION_PROMPT,
                 mode_prompts=None, model_name=MODEL_NAME, use_mongodb_checkpointer=True):
        self.core_identity = core_identity
        self.chat = chat
        self.mode_prompts = mode_prompts or MODE_SPECIFIC_PROMPTS.copy()
        self.model_name = model_name
        
        # Initialize core components
        self.llm = self._initialize_llm()
        self.vector_memory = VectorMemoryManager() if self.llm else None
        
        # Cache for AI analysis
        self.analysis_cache = LRUCache(maxsize=1000)
        
        # Initialize LangGraph checkpointer
        if use_mongodb_checkpointer:
            try:
                checkpointer_client = MongoClient(MONGODB_URI)
                checkpointer_client.admin.command('ping')
                self.checkpointer = MongoDBSaver(client=checkpointer_client, db_name=DATABASE_NAME)
                logger.info("LangGraph MongoDB checkpointer initialized")
            except Exception as e:
                logger.error(f"Failed to initialize MongoDB checkpointer: {e}")
                self.checkpointer = None
        else:
            self.checkpointer = None
        
        # Setup LangGraph workflow
        self.workflow = StateGraph(LumoState)
        self._setup_enhanced_graph()
        
        if self.checkpointer:
            self.ai_app = self.workflow.compile(checkpointer=self.checkpointer)
            logger.info("LangGraph workflow compiled with MongoDB persistence")
        else:
            logger.warning("LangGraph workflow compiled without persistence")
        
        logger.info("Enhanced Lumo Agent fully initialized")
    
    def _initialize_llm(self):
        if not GEMINI_API_KEY:
            logger.error("No API key found")
            return None
        try:
            llm = ChatGoogleGenerativeAI(
                model=self.model_name,
                temperature=0.7,
                google_api_key=GEMINI_API_KEY
            )
            test_response = llm.invoke("Hello!")
            logger.info("LLM initialized and tested successfully")
            return llm
        except Exception as e:
            logger.error(f"Error initializing LLM: {e}")
            return None
    
    def update_user_profile(self, username: str, profile: Dict[str, Any]):
        try:
            config = {"configurable": {"thread_id": f"enhanced_{username}"}}
            state = self.ai_app.get_state(config).values if self.checkpointer else {}
            
            if not state:
                state = {
                    "username": username,
                    "user_profile": {},
                    "messages": [],
                    "all_chats": [],
                    "interaction_count": 0,
                    "summaries": [],
                    "timeline_memory": {},
                    "vector_memory_metadata": [],
                    "current_mode": "general",
                    "current_emotion": "neutral",
                    "created_at": datetime.now(UTC).isoformat(),
                    "last_updated": datetime.now(UTC).isoformat(),
                    "user_timezone": "UTC"
                }
            
            state["user_profile"] = profile
            state["last_updated"] = datetime.now(UTC).isoformat()
            
            if self.checkpointer:
                self.ai_app.update_state(config, state)
                logger.info(f"Updated profile for {username} in checkpointer")
            
            self._sync_to_users_collection(state)
            return {"success": True}
        except Exception as e:
            logger.error(f"Error updating profile for {username}: {e}")
            return {"success": False, "error": str(e)}
    
    def _sync_to_users_collection(self, state: LumoState):
        try:
            username = state.get("username", "unknown")
            chats = state.get("all_chats", [])
            profile = state.get("user_profile", {})
            summaries = state.get("summaries", [])
            timeline_memory = state.get("timeline_memory", {})
            
            user_doc = {
                "username": username,
                "profile": profile,
                "chats": chats,
                "interaction_count": state.get("interaction_count", 0),
                "summaries": summaries,
                "timeline_summaries": [timeline_memory] if timeline_memory else [],
                "created_at": state.get("created_at", datetime.now(UTC).isoformat()),
                "updated_at": state.get("last_updated", datetime.now(UTC).isoformat()),
                "storage_type": "mongodb_with_checkpointer"
            }
            
            users_collection.update_one(
                {"_id": username},
                {"$set": user_doc},
                upsert=True
            )
            logger.info(f"Synced state to users collection for {username}")
        except Exception as e:
            logger.error(f"Error syncing to users collection: {e}")
    
    def get_user_info(self, username: str) -> Dict[str, Any]:
        try:
            config = {"configurable": {"thread_id": f"enhanced_{username}"}}
            state = self.ai_app.get_state(config).values if self.checkpointer else {}
            
            if state:
                return {
                    "username": state.get("username", username),
                    "profile": state.get("user_profile", {}),
                    "chats": state.get("all_chats", []),
                    "interaction_count": state.get("interaction_count", 0),
                    "summaries": state.get("summaries", []),
                    "timeline_summaries": [state.get("timeline_memory", {})] if state.get("timeline_memory") else [],
                    "created_at": state.get("created_at"),
                    "updated_at": state.get("last_updated"),
                    "status": "found",
                    "storage_type": "checkpointer"
                }
            else:
                return {
                    "username": username,
                    "profile": {},
                    "chats": [],
                    "interaction_count": 0,
                    "summaries": [],
                    "timeline_summaries": [],
                    "created_at": None,
                    "updated_at": None,
                    "status": "new_user",
                    "storage_type": "checkpointer"
                }
        except Exception as e:
            logger.error(f"Error retrieving user info for {username}: {e}")
            return {"status": "error", "error": str(e)}
    
    def delete_user_data(self, username: str) -> Dict[str, Any]:
        try:
            config = {"configurable": {"thread_id": f"enhanced_{username}"}}
            if self.checkpointer:
                checkpoints = list(self.ai_app.get_state_history(config))
                for checkpoint in checkpoints:
                    self.ai_app.delete_state(config)
                logger.info(f"Deleted checkpoints for {username}")
            
            if self.vector_memory and self.vector_memory.collection:
                self.vector_memory.collection.delete(where={"username": username})
                logger.info(f"Deleted ChromaDB memories for {username}")
            
            users_collection.delete_one({"_id": username})
            logger.info(f"Deleted users collection data for {username}")
            
            return {"success": True}
        except Exception as e:
            logger.error(f"Error deleting user data for {username}: {e}")
            return {"success": False, "error": str(e)}
    
    def _ai_analyze_intent_and_emotion(self, user_message: str) -> dict:
        if not user_message or not user_message.strip():
            return {"mode": "general", "emotion": "neutral", "confidence": 0.3, "reasoning": "Empty message"}
        
        normalized_message = user_message.lower().strip()
        if normalized_message in self.analysis_cache:
            logger.info(f"Cache hit for analysis: {normalized_message}")
            return self.analysis_cache[normalized_message]
        
        if not self.llm:
            result = {"mode": "general", "emotion": "neutral", "confidence": 0.5, "reasoning": "LLM not available"}
            self.analysis_cache[normalized_message] = result
            return result
        
        try:
            analysis_prompt = f"{INTENT_ANALYSIS_PROMPT}\n\nUser message: \"{user_message}\""
            response = self.llm.invoke(analysis_prompt)
            response_content = response.content.strip()
            if response_content.startswith('{') and response_content.endswith('}'):
                analysis_result = json.loads(response_content)
                if all(key in analysis_result for key in ["mode", "emotion"]):
                    self.analysis_cache[normalized_message] = analysis_result
                    logger.info(f"Analyzed and cached: Mode={analysis_result['mode']}, Emotion={analysis_result['emotion']}")
                    return analysis_result
            return self._fallback_analysis(user_message)
        except Exception as e:
            logger.error(f"AI analysis error: {e}")
            result = self._fallback_analysis(user_message)
            self.analysis_cache[normalized_message] = result
            return result
    
    def _fallback_analysis(self, user_message: str) -> dict:
        user_lower = user_message.lower()
        mode = "general"
        emotion = "neutral"
        
        if any(word in user_lower for word in ["play", "game", "fun", "bored"]):
            mode = "game"
        elif any(word in user_lower for word in ["story", "tell", "read"]):
            mode = "story"
        elif any(word in user_lower for word in ["learn", "how", "why", "what", "explain"]):
            mode = "learning"
        
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
        try:
            if not state.get("messages") or len(state["messages"]) == 0:
                return "general"
            
            last_message = state["messages"][-1].content.lower()
            
            if len(state["messages"]) == 1:
                if "yes" in last_message or "yeah" in last_message or "sure" in last_message:
                    return "activities"
                elif "no" in last_message or "nope" in last_message:
                    return "general"
            
            if "stories" in last_message or "story" in last_message:
                return "story"
            elif "games" in last_message or "game" in last_message or "play" in last_message:
                return "game"
            elif "learn" in last_message or "how" in last_message or "why" in last_message or "what" in last_message:
                return "learning"
            
            return "general"
        except Exception as e:
            logger.error(f"Router error: {e}")
            return "general"
    
    def _enhance_state_with_memory(self, state: LumoState) -> LumoState:
        try:
            username = state.get("username", "unknown")
            if not state.get("messages"):
                return state
            
            current_message = state["messages"][-1].content if state["messages"] else ""
            
            if self.vector_memory:
                relevant_memories = self.vector_memory.retrieve_relevant_memories(current_message, username, n_results=3)
                if relevant_memories:
                    memory_context = "RELEVANT MEMORIES:\n" + "\n".join([f"[MEMORY]: {mem}" for mem in relevant_memories])
                    state["summary_context"] = memory_context
                    logger.info(f"Enhanced state with {len(relevant_memories)} ChromaDB memories for {username}")
            
            return state
        except Exception as e:
            logger.error(f"Memory enhancement error: {e}")
            return state
    
    def _create_conversation_summary(self, username: str, messages: List[Any], start_idx: int, end_idx: int) -> Dict[str, Any]:
        try:
            if not self.llm:
                return {
                    "content": f"Basic summary: {len(messages)} messages from {start_idx} to {end_idx}",
                    "timestamp": datetime.now(UTC).isoformat(),
                    "range": f"{start_idx}-{end_idx}",
                    "type": "fallback_summary"
                }
            
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
                "timestamp": datetime.now(UTC).isoformat(),
                "range": f"{start_idx}-{end_idx}",
                "type": "ai_generated_summary",
                "message_count": end_idx - start_idx + 1
            }
        except Exception as e:
            logger.error(f"Summary creation error for {username}, range {start_idx}-{end_idx}: {e}", exc_info=True)
            return {
                "content": f"Error creating summary: {str(e)}",
                "timestamp": datetime.now(UTC).isoformat(),
                "range": f"{start_idx}-{end_idx}",
                "type": "error_summary"
            }
    
    def _process_timeline(self, username: str, messages: List[Any], range_str: str, existing_timeline: Dict[str, Any]):
        try:
            logger.info(f"Processing timeline in background for {username}")
            start_idx, end_idx = map(int, range_str.split('-'))
            temp_summary = self._create_conversation_summary(username, messages, start_idx, end_idx)
            temp_state = {
                "username": username,
                "timeline_memory": existing_timeline,
                "interaction_count": end_idx + 1,
                "vector_memory_metadata": []
            }
            updated_state = self._update_timeline_with_summary(temp_state, temp_summary)
            
            if self.vector_memory and updated_state.get("timeline_memory", {}).get("story"):
                doc_id = self.vector_memory.store_timeline_memory(
                    username=username,
                    timeline_text=updated_state["timeline_memory"]["story"],
                    metadata={
                        "updated_at": updated_state["timeline_memory"].get("updated_at"),
                        "total_interactions": updated_state["timeline_memory"].get("total_interactions")
                    }
                )
                if doc_id:
                    vector_metadata = updated_state.get("vector_memory_metadata", [])
                    existing = next((item for item in vector_metadata if item["doc_id"] == doc_id), None)
                    new_metadata = {
                        "doc_id": doc_id,
                        "timestamp": updated_state["timeline_memory"].get("updated_at"),
                        "range": temp_summary.get("range")
                    }
                    if existing:
                        vector_metadata.remove(existing)
                    vector_metadata.append(new_metadata)
                    updated_state["vector_memory_metadata"] = vector_metadata
            
            if self.checkpointer:
                config = {"configurable": {"thread_id": f"enhanced_{username}"}}
                self.ai_app.update_state(config, updated_state)
            self._sync_to_users_collection(updated_state)
            
            logger.info(f"Background timeline processing complete for {username}")
        except Exception as e:
            logger.error(f"Background timeline processing error for {username}: {e}", exc_info=True)
    
    def _update_timeline_with_summary(self, state: LumoState, temp_summary: Dict[str, Any]) -> LumoState:
        try:
            if not self.llm:
                return state
            
            username = state.get("username", "unknown")
            timeline = state.get("timeline_memory", {})
            current_time = datetime.now(UTC)
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
            
            state["timeline_memory"] = {
                "story": updated_story.strip(),
                "updated_at": current_time.isoformat(),
                "total_interactions": state.get("interaction_count", 0),
                "last_summary_range": summary_range,
                "last_summary_processed": current_time.isoformat()
            }
            
            state["summaries"] = state.get("summaries", []) + [temp_summary]
            logger.info(f"Timeline memory updated with summary {summary_range} for {username}")
            return state
        except Exception as e:
            logger.error(f"Timeline update error for {username}: {e}", exc_info=True)
            return state
    
    def _call_llm_with_enhanced_context(self, state: LumoState, interaction_type: str = "general"):
        try:
            if not self.llm:
                return {"messages": state["messages"] + [AIMessage(content="I'm having trouble thinking right now!")]}
        
            messages = state.get("messages", [])
            username = state.get("username", "unknown")
            emotion = state.get("current_emotion", "neutral")
            summary_context = state.get("summary_context", "")
            profile = state.get("user_profile", {})
            
            is_first_interaction = len(messages) == 0 or (len(messages) == 1 and isinstance(messages[0], HumanMessage))
            
            base_prompt = self._get_combined_prompt(interaction_type)
            timezone = state.get("user_timezone", "UTC")
            temporally_aware_prompt = self._add_temporal_context_to_prompt(base_prompt, timezone)
            
            enhanced_prompt = f"{temporally_aware_prompt}\n\nEMOTION: {emotion}"
            if summary_context:
                enhanced_prompt = f"{enhanced_prompt}\n\n{summary_context}"
                logger.info(f"Using enhanced timeline memory context for {username}")
            
            if is_first_interaction and profile:
                child_name = profile.get("child_name", username)
                interests = profile.get("interests", "fun things")
                topics_to_avoid = profile.get("topics_to_avoid", "none")
                enhanced_prompt += f"\n\nFIRST INTERACTION CONTEXT:\nChild Name: {child_name}\nInterests: {interests}\nTopics to Avoid: {topics_to_avoid}"
            
            conversation_text = ""
            for msg in messages[-20:]:
                if hasattr(msg, 'content'):
                    msg_type = "User" if isinstance(msg, HumanMessage) else "Assistant"
                    conversation_text += f"{msg_type}: {msg.content}\n"
            
            full_prompt = f"{enhanced_prompt}\n\nConversation:\n{conversation_text}\n\nAssistant:"
            
            response = self.llm.invoke(full_prompt)
            ai_content = response.content if hasattr(response, 'content') else str(response)
            
            return {"messages": messages + [AIMessage(content=ai_content)]}
        except Exception as e:
            logger.error(f"Error in _call_llm_with_enhanced_context: {e}")
            return {"messages": messages + [AIMessage(content="I'm having trouble thinking right now!")]}

    def _setup_enhanced_graph(self):
        def enhance_and_route(state: LumoState) -> str:
            enhanced_state = self._enhance_state_with_memory(state)
            return self._router(enhanced_state)
        
        self.workflow.add_node("general", lambda state: self._call_llm_with_enhanced_context(state, "general"))
        self.workflow.add_node("activities", lambda state: self._call_llm_with_enhanced_context(state, "activities"))
        self.workflow.add_node("game", lambda state: self._call_llm_with_enhanced_context(state, "game"))
        self.workflow.add_node("story", lambda state: self._call_llm_with_enhanced_context(state, "story"))
        self.workflow.add_node("learning", lambda state: self._call_llm_with_enhanced_context(state, "learning"))
        
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
        
        self.workflow.add_edge("general", END)
        self.workflow.add_edge("activities", END)
        self.workflow.add_edge("game", END)
        self.workflow.add_edge("story", END)
        self.workflow.add_edge("learning", END)
    
    def _get_combined_prompt(self, interaction_type: str) -> str:
        mode_prompt = self.mode_prompts.get(interaction_type, self.mode_prompts["general"])
        return f"{self.core_identity}\n\n{self.chat}\n\n{mode_prompt}"
    
    def _add_temporal_context_to_prompt(self, base_prompt: str, user_timezone: str = "UTC") -> str:
        try:
            tz = pytz.timezone(user_timezone)
            current_time = datetime.now(tz)
            time_context = f"Current time in {user_timezone}: {current_time.strftime('%Y-%m-%d %H:%M:%S %Z')}"
            return f"{base_prompt}\n\n{time_context}"
        except Exception as e:
            logger.error(f"Error adding temporal context: {e}")
            return base_prompt
    
    def process_message(self, user_message: str, username: str) -> str:
        try:
            if not self.llm:
                return "I'm having trouble thinking right now!"
            
            config = {"configurable": {"thread_id": f"enhanced_{username}"}}
            state = self.ai_app.get_state(config).values if self.checkpointer else {}
            
            if not state:
                state = {
                    "username": username,
                    "user_profile": {},
                    "messages": [],
                    "all_chats": [],
                    "interaction_count": 0,
                    "summaries": [],
                    "timeline_memory": {},
                    "vector_memory_metadata": [],
                    "current_mode": "general",
                    "current_emotion": "neutral",
                    "created_at": datetime.now(UTC).isoformat(),
                    "last_updated": datetime.now(UTC).isoformat(),
                    "user_timezone": "UTC"
                }
            
            analysis = self._ai_analyze_intent_and_emotion(user_message)
            state["current_mode"] = analysis.get("mode", "general")
            state["current_emotion"] = analysis.get("emotion", "neutral")
            
            timestamp = datetime.now(UTC).isoformat()
            chat_entry = {"timestamp": timestamp}
            
            if user_message.strip():
                state["messages"] = state.get("messages", [])[-19:] + [HumanMessage(content=user_message)]
                chat_entry["user_input"] = user_message
            
            state["interaction_count"] = state.get("interaction_count", 0) + 1
            state["last_updated"] = datetime.now(UTC).isoformat()
            
            output = self.ai_app.invoke(state, config)
            response_message = output["messages"][-1].content if output.get("messages") else "I'm having trouble responding!"
            
            state["messages"] = output["messages"][-20:]
            chat_entry["ai_response"] = response_message
            state["all_chats"] = state.get("all_chats", []) + [chat_entry]
            
            if self.checkpointer:
                self.ai_app.update_state(config, state)
            
            # Offload summary generation to a background thread every 20 interactions
            if state["interaction_count"] % 20 == 0:
                start_idx = max(0, state["interaction_count"] - 20)
                end_idx = state["interaction_count"] - 1
                range_str = f"{start_idx}-{end_idx}"
                thread = Thread(target=self._process_timeline, args=(username, state["messages"], range_str, state.get("timeline_memory", {})))
                thread.start()
                logger.info(f"Started background thread for summary generation for {username}")
            
            self._sync_to_users_collection(state)
            
            return response_message
        except Exception as e:
            logger.error(f"Error processing message: {e}", exc_info=True)
            return f"Oops, something went wrong: {str(e)}"
