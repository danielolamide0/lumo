import streamlit as st
import os
from datetime import datetime
import uuid
from ai_toy_agent import EnhancedLumoAgent, CORE_IDENTITY_PROMPT, CHAT_FOUNDATION_PROMPT, MODE_SPECIFIC_PROMPTS
from pymongo import MongoClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# MongoDB configuration
MONGODB_URI = os.getenv("MONGODB_URI")
DATABASE_NAME = os.getenv("DATABASE_NAME", "LUMO")

st.set_page_config(page_title="Lumo AI - Enhanced with Persistence", layout="wide", initial_sidebar_state="collapsed")

# Initialize session state
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if "username" not in st.session_state:
    st.session_state.username = None

if "user_info" not in st.session_state:
    st.session_state.user_info = None

if "core_identity" not in st.session_state:
    st.session_state.core_identity = CORE_IDENTITY_PROMPT

if "chat" not in st.session_state:
    st.session_state.chat = CHAT_FOUNDATION_PROMPT

if "mode_prompts" not in st.session_state:
    st.session_state.mode_prompts = MODE_SPECIFIC_PROMPTS.copy()

if "agent" not in st.session_state:
    st.session_state.agent = EnhancedLumoAgent(
        core_identity=st.session_state.core_identity,
        chat=st.session_state.chat,
        mode_prompts=st.session_state.mode_prompts,
        use_mongodb_checkpointer=True
    )
    if not st.session_state.agent.llm:
        st.error("Fatal Error: Language Model (LLM) could not be initialized. Please check your API key configuration.")
        st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

def load_user_chat_history(username: str):
    """Load user's previous chat history into Streamlit session."""
    st.session_state.messages = []
    
    if not st.session_state.get('agent') or not st.session_state.agent.checkpointer:
        print("📝 No checkpointer available, starting with empty history")
        return
    
    try:
        config = {"configurable": {"thread_id": f"enhanced_{username}"}}
        
        # Get state history from LangGraph
        state_history = list(st.session_state.agent.ai_app.get_state_history(config))
        
        if state_history:
            latest_state = state_history[0].values
            messages = latest_state.get("messages", [])
            
            # Convert LangChain messages to Streamlit format
            chat_messages = []
            for msg in messages:
                if hasattr(msg, 'content'):
                    if msg.__class__.__name__ == 'HumanMessage':
                        chat_messages.append({"role": "user", "content": msg.content})
                    elif msg.__class__.__name__ == 'AIMessage':
                        chat_messages.append({"role": "assistant", "content": msg.content})
            
            st.session_state.messages = chat_messages
            print(f"📚 Loaded {len(chat_messages)} messages from LangGraph checkpointer for {username}")
        else:
            # Try to migrate from original collection
            original_data = st.session_state.agent._load_user_data_from_original_collection(username)
            if original_data:
                messages = original_data.get("messages", [])
                chat_messages = []
                for msg in messages:
                    if hasattr(msg, 'content'):
                        if msg.__class__.__name__ == 'HumanMessage':
                            chat_messages.append({"role": "user", "content": msg.content})
                        elif msg.__class__.__name__ == 'AIMessage':
                            chat_messages.append({"role": "assistant", "content": msg.content})
                
                st.session_state.messages = chat_messages
                print(f"📚 Migrated {len(chat_messages)} messages from original collection for {username}")
            else:
                st.session_state.messages = []
                print(f"📝 No conversation history found for {username}, starting fresh")
                
    except Exception as e:
        print(f"⚠️ Error loading chat history: {e}")
        st.session_state.messages = []

def reset_conversation():
    """Reset the current conversation."""
    st.session_state.messages = []

def authenticate_user(username: str):
    """Authenticate user and load their data."""
    try:
        # Get user info using the enhanced agent
        user_info = st.session_state.agent.get_user_info(username)
        
        # Store in session
        st.session_state.username = username
        st.session_state.user_info = user_info
        st.session_state.authenticated = True
        
        # FIXED: Load user's previous chat history automatically on login
        load_user_chat_history(username)
        
        return True
        
    except Exception as e:
        st.error(f"Error during login: {str(e)}")
        return False

# 🔐 Authentication Page
if not st.session_state.authenticated:
    st.title("🧸 Welcome to Lumo AI")
    st.markdown("### Enhanced AI with Persistent Memory & Vector Search")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("---")
        st.subheader("👤 Enter Your Username")
        
        with st.form("login_form"):
            username = st.text_input(
                "Username:",
                placeholder="Enter your username",
                help="Your username will be used to save and load your conversation history."
            )
            
            submit_button = st.form_submit_button("🚀 Start Chat with Lumo!", use_container_width=True)
            
            if submit_button:
                if username.strip():
                    if authenticate_user(username.strip()):
                        st.success(f"✅ Welcome back, {username}!")
                        st.rerun()
                    else:
                        st.error("❌ Authentication failed. Please try again.")
                else:
                    st.warning("⚠️ Please enter a username to continue.")
    
    # Show system status
    with st.expander("🔍 System Status"):
        if st.session_state.get('agent'):
            st.success("✅ Enhanced AI Agent: Online")
            if st.session_state.agent.checkpointer:
                st.success("✅ LangGraph MongoDB Persistence: Active")
            else:
                st.warning("⚠️ Memory-only mode (no persistence)")
            
            if st.session_state.agent.vector_memory and st.session_state.agent.vector_memory.vector_store:
                st.success("✅ Vector Memory (ChromaDB): Active")
            else:
                st.warning("⚠️ Vector memory not available")
        else:
            st.error("❌ AI Agent: Offline")
    
    st.stop()

# 🏠 Main Application (After Authentication)
else:
    # Header
    st.title(f"🧸 Lumo AI - Enhanced Chat")
    st.markdown(f"**Logged in as:** {st.session_state.username}")
    
    # Logout button in top right
    col1, col2 = st.columns([5, 1])
    with col2:
        if st.button("🚪 Logout"):
            st.session_state.clear()
            st.rerun()

    # Sidebar with restart and debug options
    with st.sidebar:
        st.header("🔧 Lumo Controls")
        
        # Restart Lumo button
        if st.button("🔄 Restart Lumo", use_container_width=True, help="Restart Lumo with updated prompts"):
            # Clear analysis cache and reinitialize agent with current prompts
            st.session_state.agent = EnhancedLumoAgent(
                core_identity=st.session_state.core_identity,
                chat=st.session_state.chat,
                mode_prompts=st.session_state.mode_prompts,
                use_mongodb_checkpointer=True
            )
            if st.session_state.agent.llm:
                st.success("✅ Lumo restarted with updated prompts!")
                # Clear messages and reload chat history
                if st.session_state.username:
                    load_user_chat_history(st.session_state.username)
                st.rerun()
            else:
                st.error("❌ Failed to restart Lumo - LLM initialization failed")
        
        st.markdown("---")
        
        # Chat controls
        if st.button("🧹 Clear Current Chat", use_container_width=True, help="Clear chat history for this session"):
            reset_conversation()
            st.rerun()
        
        st.markdown("---")
        
        # User profile section
        st.subheader("👤 User Profile")
        if st.session_state.user_info:
            st.write(f"**Username:** {st.session_state.user_info.get('username', 'Unknown')}")
            st.write(f"**Interactions:** {st.session_state.user_info.get('interaction_count', 0)}")
            if st.session_state.user_info.get("created_at"):
                st.write(f"**Created:** {st.session_state.user_info.get('created_at')[:10]}")
            st.write(f"**Storage:** {st.session_state.user_info.get('storage_type', 'Unknown')}")
        
        # Data management
        st.subheader("🗂️ Data Management")
        if st.button("🗑️ Delete My Data", type="secondary", help="Delete all conversation data"):
            if st.checkbox("I understand this will delete all my data"):
                try:
                    result = st.session_state.agent.delete_user_data(st.session_state.username)
                    if result.get("success"):
                        st.success("Data deleted successfully!")
                        st.session_state.messages = []
                    else:
                        st.error(result.get("error", "Failed to delete data"))
                except Exception as e:
                    st.error(f"Error deleting data: {e}")

# Enhanced Configuration Section - RESTORED PROMPT REFINEMENT
with st.expander("⚙️ Configure Lumo's Architecture (Core + Chat + Specialized Modes)", expanded=False):
    
    # Create tabs for different prompt types
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🧠 Core Identity", 
        "💬 Chat (Shared)",
        "🎮 Game Mode", 
        "📚 Story Mode", 
        "🎓 Learning Mode",
        "👁️ View Combined"
    ])
    
    with tab1:
        st.subheader("Core Identity (Foundation Layer)")
        st.markdown("*Lumo's fundamental personality, safety rules, and communication style.*")
        
        core_identity_edited = st.text_area(
            "Core Identity Prompt:",
            value=st.session_state.core_identity,
            height=300,
            key="core_identity_editor",
            help="This defines Lumo's core personality and will be included in ALL interactions."
        )
        
        if st.button("🧠 Update Core Identity", key="update_core"):
            st.session_state.core_identity = core_identity_edited
            st.session_state.agent.core_identity = core_identity_edited
            st.success("Core identity updated!")
            st.info("💡 **Tip:** Click the \"🔄 Restart Lumo\" button in the sidebar to ensure changes take full effect!")
    
    with tab2:
        st.subheader("Chat Foundation (Shared Layer)")
        st.markdown("*Conversational skills shared across ALL modes.*")
        
        chat_edited = st.text_area(
            "Chat Foundation Prompt:",
            value=st.session_state.chat,
            height=300,
            key="chat_editor",
            help="This foundation is used in ALL conversation modes and handles basic conversational skills."
        )
        
        if st.button("💬 Update Chat Foundation", key="update_chat"):
            st.session_state.chat = chat_edited
            st.session_state.agent.chat = chat_edited
            st.success("Chat foundation updated!")
            st.info("💡 **Tip:** Click the \"🔄 Restart Lumo\" button in the sidebar to ensure changes take full effect!")
    
    with tab3:
        st.subheader("Game Mode (Interactive Gaming)")
        st.markdown("*Specialized behavior for playing games (builds on shared chat).*")
        
        game_prompt_edited = st.text_area(
            "Game Mode Specialization:",
            value=st.session_state.mode_prompts["game"],
            height=150,
            key="game_mode_editor"
        )

        if st.button("🎮 Update Game Mode", key="update_game"):
            st.session_state.mode_prompts["game"] = game_prompt_edited
            if hasattr(st.session_state.agent, 'mode_prompts'):
                st.session_state.agent.mode_prompts["game"] = game_prompt_edited
            st.success("Game mode updated!")
            st.info("💡 **Tip:** Click the \"🔄 Restart Lumo\" button in the sidebar to ensure changes take full effect!")
    
    with tab4:
        st.subheader("Story Mode (Interactive Storytelling)")
        st.markdown("*Specialized behavior for creating stories (builds on shared chat).*")
        
        story_prompt_edited = st.text_area(
            "Story Mode Specialization:",
            value=st.session_state.mode_prompts["story"],
            height=150,
            key="story_mode_editor"
        )
        
        if st.button("📚 Update Story Mode", key="update_story"):
            st.session_state.mode_prompts["story"] = story_prompt_edited
            if hasattr(st.session_state.agent, 'mode_prompts'):
                st.session_state.agent.mode_prompts["story"] = story_prompt_edited
            st.success("Story mode updated!")
            st.info("💡 **Tip:** Click the \"🔄 Restart Lumo\" button in the sidebar to ensure changes take full effect!")
    
    with tab5:
        st.subheader("Learning Mode (Educational Exploration)")
        st.markdown("*Specialized behavior for learning (builds on shared chat).*")
        
        learning_prompt_edited = st.text_area(
            "Learning Mode Specialization:",
            value=st.session_state.mode_prompts["learning"],
            height=150,
            key="learning_mode_editor"
        )
        
        if st.button("🎓 Update Learning Mode", key="update_learning"):
            st.session_state.mode_prompts["learning"] = learning_prompt_edited
            if hasattr(st.session_state.agent, 'mode_prompts'):
                st.session_state.agent.mode_prompts["learning"] = learning_prompt_edited
            st.success("Learning mode updated!")
            st.info("💡 **Tip:** Click the \"🔄 Restart Lumo\" button in the sidebar to ensure changes take full effect!")
    
    with tab6:
        st.subheader("Combined Architecture Preview")
        st.markdown("*See how Core Identity + Chat + Specialized Mode combine.*")
        
        mode_selector = st.selectbox(
            "Select mode to preview:",
            ["general", "game", "story", "learning"],
            key="combined_preview_selector"
        )
        
        combined_prompt = st.session_state.agent.get_combined_prompt(mode_selector)
        
        st.text_area(
            f"Combined Prompt for {mode_selector.title()} Mode:",
            value=combined_prompt,
            height=400,
            disabled=True,
            key="combined_preview"
        )
        
        st.info(f"**Total length:** {len(combined_prompt)} characters")
        st.success("**Architecture: Core Identity + Chat (Shared) + Specialized Mode**")

st.subheader("💬 Chat with Lumo")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("What do you want to say to Lumo?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner("Lumo is thinking..."):
            # Use the enhanced process_message method
            result = st.session_state.agent.process_message(st.session_state.username, prompt)
            
            if result["success"]:
                ai_response_content = result["response"]
                
                # Show enhanced info
                info_parts = []
                if result.get("persistent"):
                    info_parts.append("💾 Saved")
                else:
                    info_parts.append("⚠️ Not saved")
                
                if result.get("vector_memories", 0) > 0:
                    info_parts.append(f"🧠 {result['vector_memories']} memories")
                
                info_parts.append(f"🎭 {result.get('mode', 'general').title()}")
                info_parts.append(f"😊 {result.get('emotion', 'neutral').title()}")
                
                message_placeholder.markdown(ai_response_content)
                st.caption(" | ".join(info_parts))
            else:
                ai_response_content = result.get("response", "I encountered an error processing your message.")
                message_placeholder.markdown(ai_response_content)
                st.error("Failed to process message")
        
        st.session_state.messages.append({"role": "assistant", "content": ai_response_content}) 