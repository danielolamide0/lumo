import streamlit as st
import uuid
from ai_toy_agent import LumoAgent, CORE_IDENTITY_PROMPT, CHAT_FOUNDATION_PROMPT, MODE_SPECIFIC_PROMPTS

st.set_page_config(page_title="Lumo AI - Your Personal Companion", layout="wide", initial_sidebar_state="collapsed")

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
    st.session_state.agent = LumoAgent(
        core_identity=st.session_state.core_identity,
        chat=st.session_state.chat,
        mode_prompts=st.session_state.mode_prompts
    )
    if not st.session_state.agent.llm:
        st.error("Fatal Error: Language Model (LLM) could not be initialized. Please check your API key configuration.")
        st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

def load_user_chat_history(username: str):
    """Load user's previous chat history into Streamlit session."""
    st.session_state.messages = []
    
    if st.session_state.agent and st.session_state.agent.user_storage and st.session_state.agent.user_storage.client:
        try:
            # Get user's chat history from MongoDB
            chat_history = st.session_state.agent.user_storage.get_user_chat_history(username)
            
            # Convert to Streamlit message format
            for chat in chat_history:
                st.session_state.messages.append({"role": "user", "content": chat.get("user_input", "")})
                st.session_state.messages.append({"role": "assistant", "content": chat.get("ai_response", "")})
            
            print(f"📚 Loaded {len(chat_history)} previous chat exchanges for {username}")
            
            # If no previous history, add a simple greeting
            if not chat_history:
                with st.spinner("Lumo is greeting you..."):
                    ai_greeting = st.session_state.agent.invoke_agent("Hello!", username)
                st.session_state.messages.append({"role": "assistant", "content": ai_greeting})
                
        except Exception as e:
            print(f"Error loading chat history: {e}")
            # Fallback to greeting if error loading history
            with st.spinner("Lumo is greeting you..."):
                ai_greeting = st.session_state.agent.invoke_agent("Hello!", username)
            st.session_state.messages.append({"role": "assistant", "content": ai_greeting})
    else:
        # No persistent storage - just add greeting
        with st.spinner("Lumo is greeting you..."):
            ai_greeting = st.session_state.agent.invoke_agent("Hello!", username)
        st.session_state.messages.append({"role": "assistant", "content": ai_greeting})

def reset_conversation():
    """Reset conversation for current user."""
    st.session_state.messages = []
    if st.session_state.agent and st.session_state.agent.llm and st.session_state.username:
        with st.spinner("Lumo is greeting you..."):
            ai_greeting = st.session_state.agent.invoke_agent("Hello!", st.session_state.username)
        st.session_state.messages.append({"role": "assistant", "content": ai_greeting})

def login_user(username: str):
    """Login user and load their profile."""
    if not username or not username.strip():
        st.error("Please enter a valid username!")
        return False
    
    username = username.strip()
    
    try:
        # Get or create user in database
        user_info = st.session_state.agent.get_user_info(username)
        
        if "error" in user_info:
            if "MongoDB not available" in user_info["error"]:
                st.warning("🔄 MongoDB not available - proceeding with session-only storage")
                user_info = {
                    "username": username,
                    "total_chats": 0,
                    "storage_type": "Session-only",
                    "persistent": False
                }
            else:
                st.error(f"Error accessing user profile: {user_info['error']}")
                return False
        
        # Store in session
        st.session_state.username = username
        st.session_state.user_info = user_info
        st.session_state.authenticated = True
        
        # Load user's previous chat history instead of resetting
        load_user_chat_history(username)
        
        return True
        
    except Exception as e:
        st.error(f"Error during login: {str(e)}")
        return False

# 🔐 Authentication Page
if not st.session_state.authenticated:
    st.title("🧸 Welcome to Lumo AI")
    st.markdown("### Your Personal AI Companion")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("---")
        st.subheader("👤 Enter Your Username")
        
        with st.form("login_form"):
            username_input = st.text_input(
                "Username:",
                placeholder="Enter your name (e.g., Alice, John, etc.)",
                help="This will be used to save and load your conversation history"
            )
            
            submit_button = st.form_submit_button("🚀 Start Chatting with Lumo!", use_container_width=True)
            
            if submit_button:
                if login_user(username_input):
                    st.success(f"Welcome, {st.session_state.username}! 🎉")
                    st.rerun()
        
        st.markdown("---")

# 🏠 Main Application (After Authentication)
else:
    # Header with user info
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.title("🧸 Lumo AI - Personal Companion")
    
    with col2:
        st.markdown(f"**👤 User:** {st.session_state.username}")
        if st.session_state.user_info:
            st.markdown(f"**💬 Total Chats:** {st.session_state.user_info.get('total_chats', 0)}")
    
    with col3:
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.username = None
            st.session_state.user_info = None
            st.session_state.messages = []
            st.rerun()

    # Enhanced Configuration Section
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
            
            if st.button("✨ Update Core Identity", key="update_core"):
                st.session_state.core_identity = core_identity_edited
                st.session_state.agent.update_core_identity(core_identity_edited)
                st.success("Core identity updated! This affects all interaction modes.")
        
        with tab2:
            st.subheader("Chat (Shared Across ALL Modes)")
            st.markdown("*Conversational abilities and emotional intelligence shared across ALL modes.*")
            
            chat_edited = st.text_area(
                "Chat Foundation Prompt:",
                value=st.session_state.chat,
                height=300,
                key="chat_editor",
                help="This defines how Lumo converses and will be included in ALL interaction modes."
            )
            
            if st.button("💬 Update Chat", key="update_chat_shared"):
                st.session_state.chat = chat_edited
                st.session_state.agent.chat = chat_edited
                st.success("Chat updated! This affects all interaction modes.")
        
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
                st.session_state.agent.update_mode_prompt("game", game_prompt_edited)
                st.success("Game mode updated!")
        
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
                st.session_state.agent.update_mode_prompt("story", story_prompt_edited)
                st.success("Story mode updated!")
        
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
                st.session_state.agent.update_mode_prompt("learning", learning_prompt_edited)
                st.success("Learning mode updated!")
        
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
                ai_response_content = st.session_state.agent.invoke_agent(
                    prompt,
                    st.session_state.username  # Use username instead of conversation_id
                )
                message_placeholder.markdown(ai_response_content)
        
        st.session_state.messages.append({"role": "assistant", "content": ai_response_content})

    # Sidebar controls
    with st.sidebar:
        st.header("🎛️ Controls")
        
        if st.button("🧹 Clear Current Chat", use_container_width=True):
            reset_conversation()
            st.rerun()
        
        if st.button("🔄 Restart Lumo", use_container_width=True):
            st.session_state.agent = LumoAgent(
                core_identity=st.session_state.core_identity,
                chat=st.session_state.chat,
                mode_prompts=st.session_state.mode_prompts
            )
            if not st.session_state.agent.llm:
                st.error("Failed to re-initialize the Language Model after reset.")
            else:
                reset_conversation()
                st.rerun()
        
        st.markdown("---")
        st.subheader("👤 User Profile")
        if st.session_state.user_info:
            st.write(f"**Username:** {st.session_state.user_info.get('username', 'Unknown')}")
            st.write(f"**Total Conversations:** {st.session_state.user_info.get('total_chats', 0)}")
            st.write(f"**Storage:** {st.session_state.user_info.get('storage_type', 'Unknown')}")
            
            created_at = st.session_state.user_info.get('created_at')
            if created_at:
                st.write(f"**Member Since:** {created_at.strftime('%Y-%m-%d') if hasattr(created_at, 'strftime') else str(created_at)}")