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
import logging
import asyncio

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    logger.info("Successfully imported MongoDBSaver")
    MONGODB_CHECKPOINTER_AVAILABLE = True
except ImportError as e:
    logger.warning(f"MongoDB checkpointer import error: {e}")
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
        logger.info("Using SQLite checkpointer as fallback")
        MongoDBSaver = SqliteSaver
        MONGODB_CHECKPOINTER_AVAILABLE = True
    except ImportError as e:
        logger.warning(f"SQLite checkpointer import error: {e}")
        MongoDBSaver = None
        MONGODB_CHECKPOINTER_AVAILABLE = False

# --- Configuration ---
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-2.5-flash-preview-04-17")
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
MONGODB_URI = os.getenv("MONGODB_URI")
DATABASE_NAME = os.getenv("DATABASE_NAME", "LUMO")

# Configuration loading with proper fallback chain
try:
    if not GEMINI_API_KEY:
        GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY")
        MODEL_NAME = st.secrets.get("MODEL_NAME", MODEL_NAME)
        MONGODB_URI = st.secrets.get("MONGODB_URI")
        DATABASE_NAME = st.secrets.get("DATABASE_NAME", DATABASE_NAME)
except Exception as e:
    logger.warning(f"Could not load API configuration: {e}")

logger.info(f"Configuration loaded: MODEL_NAME={MODEL_NAME}, DATABASE_NAME={DATABASE_NAME}")

# Test MongoDB connection
try:
    if MONGODB_URI:
        client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
        client.server_info()
        logger.info(f"Successfully connected to MongoDB at {MONGODB_URI}")
        db = client[DATABASE_NAME]
        logger.info(f"Connected to database: {DATABASE_NAME}")
    else:
        logger.warning("No MongoDB URI provided, will use SQLite fallback")
except Exception as e:
    logger.error(f"MongoDB connection error: {e}")
    logger.warning("Falling back to SQLite")
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
        logger.info("SQLite database initialized")
    except Exception as e:
        logger.error(f"SQLite initialization error: {e}")

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
    messages: List[Any]
    username: str
    user_profile: Dict[str, Any]
    user_timezone: Optional[str]
    interaction_count: int
    timeline_memory: Dict[str, Any]
    vector_memory_metadata: Dict[str, Any]
    current_mode: str
    current_emotion: str
    summary_context: Optional[str]
    created_at: str
    last_updated: str

class VectorMemoryManager:
    def __init__(self, collection_name: str = "lumo_timeline_memory"):
        self.collection_name = collection_name
        self.vector_store = None
        self._initialized = False
        self._initialize_vector_store()
    
    def _initialize_vector_store(self):
        if self._initialized:
            return
        try:
            logger.info("Initializing vector store with HuggingFace embeddings...")
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
            self._initialized = True
            logger.info("Vector memory initialized for timeline summaries")
        except Exception as e:
            logger.error(f"Vector memory initialization failed: {e}")
    
    def store_timeline_memory(self, username: str, timeline: Dict[str, Any]):
        if not self._initialized or not self.vector_store:
            logger.warning("Vector store not initialized, skipping timeline storage")
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
            logger.info(f"Stored timeline memory in vector DB for {username}")
        except Exception as e:
            logger.error(f"Error storing timeline in vector DB: {e}")
    
    def retrieve_relevant_memories(self, username: str, query: str, k: int = 3) -> List[str]:
        if not self.vector_store:
            return []
        try:
            docs = self.vector_store.similarity_search(
                query,
                k=k,
                filter={"$and": [{"username": username}, {"type": "timeline_memory"}]}
            )
            memories = [f"[TIMELINE]: {doc.page_content}" for doc in docs]
            return memories
        except Exception as e:
            logger.error(f"Error retrieving memories: {e}")
            return []
    
    def get_user_timeline_count(self, username: str) -> int:
        if not self.vector_store:
            return 0
        try:
            docs = self.vector_store.similarity_search(
                "",
                k=100,
                filter={"$and": [{"username": username}, {"type": "timeline_memory"}]}
            )
            return len(docs)
        except Exception:
            return 0

class EnhancedLumoAgent:
    def __init__(self, core_identity=CORE_IDENTITY_PROMPT, chat=CHAT_FOUNDATION_PROMPT,
                 mode_prompts=None, model_name=MODEL_NAME, use_mongodb_checkpointer=True):
        self.core_identity = core_identity
        self.chat = chat
        self.mode_prompts = mode_prompts or MODE_SPECIFIC_PROMPTS.copy()
        self.model_name = model_name
        
        # Initialize core components
        self.llm = self._initialize_llm()
        self.vector_memory = VectorMemoryManager()
        
        # Cache for AI analysis
        self._analysis_cache = {}
        
        # Initialize MongoDB client
        self.mongo_client = None
        self.db = None
        self.users_collection = None
        if use_mongodb_checkpointer:
            try:
                self.mongo_client = MongoClient(MONGODB_URI)
                self.db = self.mongo_client[DATABASE_NAME]
                self.users_collection = self.db.users
            except Exception as e:
                logger.error(f"Error connecting to MongoDB: {e}")
        
        # Initialize LangGraph checkpointer
        if MONGODB_CHECKPOINTER_AVAILABLE:
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
            self.ai_app = self.workflow.compile()
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

    def _ai_analyze_intent_and_emotion(self, user_message: str) -> dict:
        if not user_message or not user_message.strip():
            return {"mode": "general", "emotion": "neutral", "confidence": 0.3, "reasoning": "Empty message"}
        
        if user_message in self._analysis_cache:
            return self._analysis_cache[user_message]
        
        if not self.llm:
            result = {"mode": "general", "emotion": "neutral", "confidence": 0.5, "reasoning": "LLM not available"}
            self._analysis_cache[user_message] = result
            return result
        
        try:
            analysis_prompt = f"{INTENT_ANALYSIS_PROMPT}\n\nUser message: \"{user_message}\""
            response = self.llm.invoke(analysis_prompt)
            response_content = response.content.strip()
            if response_content.startswith('{') and response_content.endswith('}'):
                analysis_result = json.loads(response_content)
                if all(key in analysis_result for key in ["mode", "emotion"]):
                    self._analysis_cache[user_message] = analysis_result
                    logger.info(f"AI Analysis: Mode={analysis_result['mode']}, Emotion={analysis_result['emotion']}")
                    return analysis_result
            return self._fallback_analysis(user_message)
        except Exception as e:
            logger.error(f"AI analysis error: {e}")
            result = self._fallback_analysis(user_message)
            self._analysis_cache[user_message] = result
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
            
            if self.vector_memory and self.vector_memory.vector_store:
                relevant_memories = self.vector_memory.retrieve_relevant_memories(
                    username, current_message, k=3
                )
                if relevant_memories:
                    memory_context = "RELEVANT MEMORIES:\n" + "\n".join(relevant_memories)
                    state["summary_context"] = memory_context
                    logger.info(f"Enhanced state with {len(relevant_memories)} relevant memories")
            
            return state
        except Exception as e:
            logger.error(f"Memory enhancement error: {e}")
            return state

    def _create_conversation_summary(self, username: str, messages: List[Any], start_idx: int, end_idx: int) -> Dict[str, Any]:
        try:
            if not self.llm:
                return {
                    "content": f"Basic summary: {len(messages)} messages from {start_idx} to {end_idx}",
                    "timestamp": datetime.utcnow().isoformat(),
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
                "timestamp": datetime.utcnow().isoformat(),
                "range": f"{start_idx}-{end_idx}",
                "type": "ai_generated_summary",
                "message_count": end_idx - start_idx + 1
            }
        except Exception as e:
            logger.error(f"Summary creation error: {e}")
            return {
                "content": f"Error creating summary: {str(e)}",
                "timestamp": datetime.utcnow().isoformat(),
                "range": f"{start_idx}-{end_idx}",
                "type": "error_summary"
            }

    def _store_memories_in_vector_db(self, state: LumoState):
        if not self.vector_memory:
            return
        username = state.get("username", "unknown")
        timeline = state.get("timeline_memory", {})
        if timeline and timeline.get("story"):
            self.vector_memory.store_timeline_memory(username, timeline)

    async def _process_timeline_async(self, username: str, messages: List[Any], range_str: str, existing_timeline: Dict[str, Any]):
        try:
            logger.info(f"Processing timeline in background for {username}")
            start_idx, end_idx = map(int, range_str.split('-'))
            temp_summary = self._create_conversation_summary(username, messages, start_idx, end_idx)
            temp_state = {
                "username": username,
                "timeline_memory": existing_timeline,
                "interaction_count": end_idx + 1
            }
            updated_state = self._update_timeline_with_summary(temp_state, temp_summary)
            updated_timeline = updated_state.get("timeline_memory", {})
            if updated_timeline and self.vector_memory:
                self.vector_memory.store_timeline_memory(username, updated_timeline)
                logger.info(f"Background timeline processing complete for {username}")
            return updated_state
        except Exception as e:
            logger.error(f"Background timeline processing error: {e}")

    def _update_timeline_with_summary(self, state: LumoState, temp_summary: Dict[str, Any]) -> LumoState:
        try:
            if not self.llm:
                return state
            
            username = state.get("username", "unknown")
            timeline = state.get("timeline_memory", {})
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
            
            state["timeline_memory"] = {
                "story": updated_story.strip(),
                "updated_at": current_time.isoformat(),
                "total_interactions": state.get("interaction_count", 0),
                "last_summary_range": summary_range,
                "last_summary_processed": current_time.isoformat()
            }
            
            logger.info(f"Timeline memory updated with summary {summary_range} for {username}")
            return state
        except Exception as e:
            logger.error(f"Timeline update error: {e}")
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
            
            is_first_interaction = len(messages) == 1 and isinstance(messages[0], HumanMessage)
            
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
                enhanced_prompt += f"\n\nFIRST INTERACTION CONTEXT:\nChild Name: {child_name}\nInterests: {interests}"
            
            conversation_text = ""
            for msg in messages:
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
            logger.error(f"Temporal context error: {e}")
            return base_prompt

    def _sync_to_users_collection(self, state: LumoState):
        if self.users_collection is None:
            return
        try:
            username = state["username"]
            current_time = datetime.now(pytz.UTC)
            
            message_history = []
            for i, msg in enumerate(state["messages"]):
                if hasattr(msg, 'content'):
                    if isinstance(msg, HumanMessage):
                        msg_dict = {
                            "user_input": msg.content,
                            "timestamp": current_time.isoformat(),
                            "interaction_id": i
                        }
                    elif isinstance(msg, AIMessage):
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
            
            timeline_summaries = {}
            timeline_data = state.get("timeline_memory", {})
            if timeline_data and timeline_data.get("story"):
                timeline_summaries = {
                    "story": timeline_data.get("story", ""),
                    "created_at": timeline_data.get("updated_at", current_time.isoformat()),
                    "updated_at": timeline_data.get("updated_at", current_time.isoformat()),
                    "summaries_processed": timeline_data.get("summaries_processed", 0) + 1,
                    "last_interaction_time": current_time.isoformat(),
                    "first_interaction_time": state.get("created_at", current_time.isoformat()),
                    "total_interactions": state.get("interaction_count", 0)
                }
            
            user_doc = {
                "_id": username,  # FIX: Use username as _id
                "username": username,
                "chats": message_history,
                "profile": state.get("user_profile", {}),
                "interaction_count": state.get("interaction_count", 0),
                "timeline_summaries": timeline_summaries,
                "summaries": state.get("summaries", []),  # FIX: Include summaries
                "current_mode": state.get("current_mode", "general"),
                "current_emotion": state.get("current_emotion", "neutral"),
                "created_at": state.get("created_at", current_time.isoformat()),
                "updated_at": current_time.isoformat(),
                "email": f"{username}@lumo.ai",
                "user_timezone": state.get("user_timezone", "UTC"),
                "vector_memory_metadata": state.get("vector_memory_metadata", {}),
                "storage_notes": {
                    "primary": "LangGraph MongoDB Checkpointer",
                    "secondary": "Users Collection",
                    "vector_memory": "ChromaDB Timeline Summaries",
                    "format": "Compatible with original users collection schema"
                }
            }
            
            result = self.users_collection.replace_one({"_id": username}, user_doc, upsert=True)
            logger.info(f"{'Created' if result.upserted_id else 'Updated'} user record for {username}")
        except Exception as e:
            logger.error(f"Error syncing to users collection: {e}")

    def process_message(self, message: str, username: str) -> str:
        try:
            # Get user info or create new user
            user_info = self.get_user_info(username)
            if not user_info or user_info.get("status") == "new_user":
                user_info = {
                    "_id": username,  # FIX: Ensure _id is username
                    "username": username,
                    "interaction_count": 0,
                    "created_at": datetime.utcnow(),
                    "storage_type": "mongodb",
                    "profile": {},  # Initialize empty profile
                    "summaries": []  # Initialize summaries
                }
                self.users_collection.insert_one(user_info)
            
            # Initialize state
            config = {"configurable": {"thread_id": f"enhanced_{username}"}}
            state_history = list(self.ai_app.get_state_history(config))
            if state_history:
                state = state_history[0].values
            else:
                created_at = user_info.get("created_at")
                if isinstance(created_at, str):
                    try:
                        created_at = datetime.fromisoformat(created_at)
                    except ValueError:
                        created_at = datetime.utcnow()
                elif not isinstance(created_at, datetime):
                    created_at = datetime.utcnow()
                
                state = {
                    "messages": [],
                    "username": username,
                    "user_profile": user_info.get("profile", {}),
                    "user_timezone": "UTC",
                    "interaction_count": user_info.get("interaction_count", 0),
                    "timeline_memory": user_info.get("timeline_summaries", {}),
                    "vector_memory_metadata": {},
                    "current_mode": "general",
                    "current_emotion": "neutral",
                    "summary_context": None,
                    "created_at": created_at.isoformat(),
                    "last_updated": datetime.utcnow().isoformat(),
                    "summaries": user_info.get("summaries", [])  # FIX: Include summaries
                }
            
            # Load chat history from MongoDB if checkpointer is empty
            if not state["messages"] and user_info.get("chats"):
                for chat in user_info["chats"]:
                    if "user_input" in chat:
                        state["messages"].append(HumanMessage(content=chat["user_input"]))
                    if "ai_response" in chat:
                        state["messages"].append(AIMessage(content=chat["ai_response"]))
            
            # Add current message
            state["messages"].append(HumanMessage(content=message))
            
            # Update mode and emotion
            analysis = self._ai_analyze_intent_and_emotion(message)
            state["current_mode"] = analysis["mode"]
            state["current_emotion"] = analysis["emotion"]
            
            # Process through LangGraph
            response_state = self.ai_app.invoke(state, config=config)
            response = response_state["messages"][-1].content
            
            # Update interaction count
            state["interaction_count"] += 1
            
            # Generate summary if at 20 interactions
            if state["interaction_count"] % 20 == 0:
                start_idx = max(0, state["interaction_count"] - 20)
                end_idx = state["interaction_count"] - 1
                loop = asyncio.get_event_loop()
                updated_state = loop.run_until_complete(
                    self._process_timeline_async(username, state["messages"], f"{start_idx}-{end_idx}", state["timeline_memory"])
                )
                state["timeline_memory"] = updated_state["timeline_memory"]
                # FIX: Store summary in summaries array
                summary = self._create_conversation_summary(username, state["messages"], start_idx, end_idx)
                state["summaries"] = state.get("summaries", []) + [summary]
            
            # Sync to MongoDB
            self._sync_to_users_collection(state)
            
            return response
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return "I'm having trouble processing your message right now. Please try again!"

    def get_user_info(self, username: str) -> dict:
        if not self.checkpointer:
            return {"error": "Checkpointer not available"}
        try:
            config = {"configurable": {"thread_id": f"enhanced_{username}"}}
            result = {
                "username": username,
                "storage_sources": [],
                "errors": []
            }
            
            # Query with _id as username
            users_data = self.users_collection.find_one(
                {"_id": username},
                {
                    "profile": 1,
                    "interaction_count": 1,
                    "created_at": 1,
                    "updated_at": 1,
                    "chats": 1,
                    "timeline_summaries": 1,
                    "summaries": 1
                }
            )
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
            
            if result["storage_sources"]:
                result["status"] = "found"
                result["persistent"] = True
                result["primary_storage"] = "Users Collection" if users_data else result["storage_sources"][0]
            else:
                result["status"] = "new_user"
                result["persistent"] = bool(self.checkpointer)
            
            if self.vector_memory:
                result["vector_memory_count"] = self.vector_memory.get_user_timeline_count(username)
            
            return result
        except Exception as e:
            return {"error": f"Failed to get user info: {str(e)}", "username": username}

    def delete_user_data(self, username: str) -> dict:
        if not self.checkpointer:
            return {"error": "Checkpointer not available"}
        try:
            config = {"configurable": {"thread_id": f"enhanced_{username}"}}
            deleted_count = 0
            if hasattr(self.checkpointer, 'client'):
                mongo_client = self.checkpointer.client
                db = mongo_client[DATABASE_NAME]
                checkpoints_collection = db["lumo_checkpoints"]
                result = checkpoints_collection.delete_many({"thread_id": f"enhanced_{username}"})
                deleted_count = result.deleted_count
            
            deleted_vectors = 0
            if self.vector_memory and self.vector_memory.vector_store:
                try:
                    docs = self.vector_memory.vector_store.similarity_search(
                        "", k=100, filter={"username": username}
                    )
                    if docs:
                        doc_ids = [doc.metadata.get("id") for doc in docs if doc.metadata.get("id")]
                        if doc_ids:
                            self.vector_memory.vector_store.delete(ids=doc_ids)
                            deleted_vectors = len(doc_ids)
                except Exception as ve:
                    logger.warning(f"Vector deletion warning: {ve}")
            
            return {
                "success": True,
                "username": username,
                "deleted_checkpoints": deleted_count,
                "deleted_vectors": deleted_vectors
            }
        except Exception as e:
            return {"error": f"Failed to delete user data: {str(e)}"}

    def get_combined_prompt(self, mode: str = "general", user_context: str = "", memory_context: str = "") -> str:
        mode_prompt = self.mode_prompts.get(mode, self.mode_prompts["general"])
        context_section = ""
        if memory_context:
            context_section += f"\n\n=== RELEVANT MEMORIES ===\n{memory_context}\n"
        if user_context:
            context_section += f"\n=== USER CONTEXT ===\n{user_context}\n"
        return f"{self.core_identity}\n\n{self.chat}\n\n{mode_prompt}{context_section}"

if __name__ == "__main__":
    logger.info("Initializing Enhanced Lumo Agent for testing...")
    agent = EnhancedLumoAgent()
    if not agent.llm:
        logger.error("LLM could not be initialized. Exiting.")
    else:
        logger.info("Lumo is ready! (Type 'quit' to end)")
        username = "test_user"
        print(f"Lumo: Hi there! I'm Lumo, your friendly AI companion!")
        while True:
            user_input = input("You: ")
            if user_input.lower() == 'quit':
                print("Lumo: Bye bye for now! It was fun chatting with you!")
                break
            if not user_input.strip():
                continue
            ai_response = agent.process_message(user_input, username)
            print(f"Lumo: {ai_response}")
