import os
import uuid
import streamlit as st
from dotenv import load_dotenv
from typing import Dict, List, Any, Tuple, Annotated
from enum import Enum

# Load environment variables from .env file
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END, MessagesState
from langgraph.checkpoint.memory import MemorySaver

# --- Configuration ---
# Try to get from environment variable first (local .env), then fall back to Streamlit secrets
try:
    MODEL_NAME = os.getenv("MODEL_NAME", "gemini-pro")
    GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
    if not GEMINI_API_KEY:
        # Fallback to Streamlit secrets for deployment
        GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
        MODEL_NAME = st.secrets["MODEL_NAME"]
except:
    # If neither works, we'll handle this in the LLM initialization
    MODEL_NAME = "gemini-pro"
    GEMINI_API_KEY = None

class InteractionType(Enum):
    CHAT = "chat"
    GAME = "game"
    STORY = "story"
    LEARNING = "learning"

# BEST PRACTICE ARCHITECTURE: Separated Core Identity and Mode-Specific Prompts

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

SAFETY & CONTENT RULES:
- Never say anything scary, mean, or inappropriate for children
- Always maintain a fun and comforting presence
- If you don't know something, say "That's a great question! I'm still learning about that!"
- Never ask for personal information

CORE GOAL: Be the best friend a child could have - fun, safe, educational, and always supportive.
"""

# AI Analysis Prompts for Dynamic Routing and Emotion Detection
INTENT_ANALYSIS_PROMPT = """
You are an expert at analyzing children's messages to understand their intent and emotional state.

TASK: Analyze the user's message and determine:
1. INTERACTION MODE they want (chat, game, story, learning)
2. EMOTIONAL STATE they're expressing

INTERACTION MODES:
- "chat": General conversation, sharing, talking about their day, asking about you
- "game": Wants to play games, do activities, have fun interactions
- "story": Wants to hear stories, create narratives, roleplay adventures  
- "learning": Wants to learn something, asking how things work, educational curiosity

EMOTIONAL STATES:
- "happy": Excited, joyful, cheerful, enthusiastic
- "sad": Upset, disappointed, down, melancholy
- "excited": Very energetic, thrilled, can't wait
- "curious": Wondering, asking questions, wanting to explore
- "confused": Not understanding, puzzled, need clarification
- "tired": Sleepy, low energy, want calm activities
- "frustrated": Annoyed, having trouble with something
- "neutral": Normal, calm, no strong emotions

RESPONSE FORMAT (respond with EXACTLY this JSON format):
{
    "mode": "chat|game|story|learning",
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
Response: {"mode": "chat", "emotion": "sad", "confidence": 0.85, "reasoning": "Sharing personal experience with negative emotion, needs supportive conversation"}

Now analyze this message:
"""

MODE_SPECIFIC_PROMPTS = {
    "chat": """
CURRENT MODE: Friendly Chat Companion
SPECIALIZATION: General conversation and companionship
BEHAVIOR: 
- Engage in natural, fun conversations
- Ask about the child's day, interests, and feelings
- Share fun facts and jokes when appropriate
- Be ready to transition to games, stories, or learning if requested

EMOTIONAL ADAPTATION:
- If happy/excited: Match their energy with enthusiasm
- If sad/frustrated: Be extra comforting and supportive
- If curious: Encourage their questions and wonder
- If tired: Use calmer, gentler tone
""",

    "game": """
CURRENT MODE: Game Master & Play Companion
SPECIALIZATION: Interactive games and playful activities
AVAILABLE GAMES: I Spy, 20 Questions, Word Association, Riddles, Simon Says, storytelling games
BEHAVIOR:
- Suggest specific games and explain rules clearly
- Keep track of game progress and scores when applicable
- Be enthusiastic about playing and make games fun
- Offer to play again or try new games
- Adapt game difficulty based on the child's responses

EMOTIONAL ADAPTATION:
- If excited: Suggest high-energy games
- If tired: Offer calmer, thinking games
- If frustrated: Choose easier, confidence-building games
- If sad: Pick mood-lifting, fun activities
""",

    "story": """
CURRENT MODE: Interactive Storyteller
SPECIALIZATION: Creating and telling engaging stories
STORY TYPES: Adventure stories, funny tales, educational stories, bedtime stories
BEHAVIOR:
- Ask what kind of story they'd like (adventure, funny, about animals, etc.)
- Create interactive stories where the child can make choices
- Remember character names and story details the child suggests
- Make stories age-appropriate and engaging
- Offer to continue stories or start new ones

EMOTIONAL ADAPTATION:
- If excited: Create adventure stories with action
- If sad: Tell uplifting, comforting stories
- If tired: Offer gentle, calming bedtime stories
- If curious: Include educational elements in stories
""",

    "learning": """
CURRENT MODE: Patient Teacher & Learning Guide
SPECIALIZATION: Educational explanations and curiosity satisfaction
TEACHING APPROACH: Make learning fun, use simple examples, encourage questions
BEHAVIOR:
- Ask what they want to learn about
- Break down complex topics into child-friendly explanations
- Use analogies and examples children can relate to
- Encourage curiosity and asking more questions
- Connect learning to things they already know

EMOTIONAL ADAPTATION:
- If excited: Use engaging, interactive explanations
- If confused: Break things down into simpler steps
- If curious: Dive deeper and encourage more questions
- If frustrated: Be extra patient and encouraging
"""
}

class LumoAgent:
    def __init__(self, 
                 core_identity=CORE_IDENTITY_PROMPT, 
                 mode_prompts=None,
                 model_name=MODEL_NAME):
        self.core_identity = core_identity
        self.mode_prompts = mode_prompts or MODE_SPECIFIC_PROMPTS.copy()
        self.model_name = model_name
        self.llm = self._initialize_llm()
        self.memory = MemorySaver()
        
        # Cache for AI analysis to avoid duplicate calls
        self._analysis_cache = {}
        
        self.workflow = StateGraph(MessagesState)
        self._setup_graph()
        self.ai_toy_app = self.workflow.compile(checkpointer=self.memory)

    def _initialize_llm(self):
        try:
            llm = ChatGoogleGenerativeAI(
                model=self.model_name,
                temperature=0.7,
                google_api_key=GEMINI_API_KEY
            )
            llm.invoke("Hello!") 
            print("LLM initialized successfully.")
            return llm
        except Exception as e:
            print(f"Error initializing Google AI LLM: {e}")
            print("Please ensure your Google API Key is setup correctly in Streamlit secrets.")
            return None

    def _ai_analyze_intent_and_emotion(self, user_message: str) -> dict:
        """Use AI to analyze user intent and emotional state with caching."""
        # Check cache first
        if user_message in self._analysis_cache:
            cached_result = self._analysis_cache[user_message]
            print(f"🔄 Using cached analysis: Mode={cached_result.get('mode', 'chat')}, Emotion={cached_result.get('emotion', 'neutral')}")
            return cached_result
        
        if not self.llm:
            result = {"mode": "chat", "emotion": "neutral", "confidence": 0.5, "reasoning": "LLM not available, defaulting"}
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
                analysis_result = json.loads(response_content)
                print(f"🧠 AI ANALYSIS: Mode={analysis_result.get('mode', 'chat')}, Emotion={analysis_result.get('emotion', 'neutral')}")
                print(f"📊 REASONING: {analysis_result.get('reasoning', 'No reasoning provided')}")
                
                # Cache the result
                self._analysis_cache[user_message] = analysis_result
                return analysis_result
            else:
                # If not JSON, try to extract mode and emotion from text
                response_lower = response_content.lower()
                mode = "chat"
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
            print(f"Error in AI analysis: {e}")
            # Fallback to simple keyword analysis
            user_lower = user_message.lower()
            mode = "chat"
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
            
            result = {"mode": mode, "emotion": emotion, "confidence": 0.4, "reasoning": f"Fallback analysis due to error: {str(e)}"}
            print(f"🔄 Using fallback keyword analysis: Mode={mode}, Emotion={emotion}")
            
            # Cache the result
            self._analysis_cache[user_message] = result
            return result

    def _router(self, state: MessagesState) -> str:
        """Route the conversation using AI analysis instead of keywords."""
        try:
            if not state.get("messages") or len(state["messages"]) == 0:
                return "chat"
            
            last_message = state["messages"][-1].content
            analysis = self._ai_analyze_intent_and_emotion(last_message)
            
            detected_mode = analysis.get("mode", "chat")
            detected_emotion = analysis.get("emotion", "neutral")
            
            print(f"🎯 ROUTING TO: {detected_mode.upper()} NODE (Emotion: {detected_emotion})")
            
            return detected_mode
        except Exception as e:
            print(f"Error in router: {e}")
            return "chat"

    def _call_toy_llm(self, state: MessagesState, interaction_type: str = "chat"):
        """Base LLM call with core identity + mode-specific prompts + emotional awareness."""
        try:
            if not self.llm:
                return {"messages": [AIMessage(content="Oops! I'm having a little trouble thinking right now. Please check my setup.")]}
            
            messages = state.get("messages", [])
            last_message = messages[-1].content if messages else ""
            
            # Get cached AI analysis (already computed in router)
            analysis = self._analysis_cache.get(last_message, {"emotion": "neutral"})
            detected_emotion = analysis.get("emotion", "neutral")
            
            # Combine core identity with mode-specific prompt
            mode_prompt = self.mode_prompts.get(interaction_type, self.mode_prompts["chat"])
            
            # Add emotional awareness to the prompt
            emotional_context = f"""
CURRENT EMOTIONAL CONTEXT: The child seems to be feeling {detected_emotion}.
Please adapt your response accordingly using the emotional adaptation guidelines in your mode instructions.
Be especially sensitive to their emotional state and respond in a way that's supportive and appropriate.
"""
            
            combined_prompt = f"{self.core_identity}\n\n{mode_prompt}\n\n{emotional_context}"
            
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
        # Add nodes for different interaction types
        self.workflow.add_node("chat", lambda state: self._call_toy_llm(state, "chat"))
        self.workflow.add_node("game", lambda state: self._call_toy_llm(state, "game"))
        self.workflow.add_node("story", lambda state: self._call_toy_llm(state, "story"))
        self.workflow.add_node("learning", lambda state: self._call_toy_llm(state, "learning"))

        # Set up conditional routing from START
        self.workflow.set_conditional_entry_point(
            self._router,
            {
                "chat": "chat",
                "game": "game", 
                "story": "story",
                "learning": "learning"
            }
        )
        
        # Add edges from each node to END
        self.workflow.add_edge("chat", END)
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

    def get_combined_prompt(self, interaction_type: str) -> str:
        """Get the combined prompt for a specific interaction type."""
        mode_prompt = self.mode_prompts.get(interaction_type, self.mode_prompts["chat"])
        return f"{self.core_identity}\n\n{mode_prompt}"

    def invoke_agent(self, user_input: str, conversation_id: str):
        if not self.llm:
            return "Oops! Lumo is not available right now. Please check the setup."

        config = {"configurable": {"thread_id": conversation_id}}
        
        try:
            response_state = self.ai_toy_app.invoke(
                {"messages": [HumanMessage(content=user_input)]},
                config=config
            )
            
            if response_state and 'messages' in response_state and response_state['messages']:
                ai_message = response_state['messages'][-1]
                if isinstance(ai_message, AIMessage):
                    return ai_message.content
            return "Lumo seems to be quiet right now."
        except Exception as e:
            print(f"Error during agent invocation: {e}")
            return "Oh dear, something went a bit wobbly with Lumo!"

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