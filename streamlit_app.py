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

def check_user_chat_count(username: str) -> int:
    """Check the number of chats a user has in the database."""
    try:
        mongo_client = MongoClient(MONGODB_URI)
        db = mongo_client[DATABASE_NAME]
        user = db.users.find_one({"username": username})
        if user and "chats" in user:
            return len(user["chats"])
        return 0
    except Exception as e:
        print(f"Error checking chat count: {e}")
        return 0

def authenticate_user(username: str):
    """Authenticate user and load their data."""
    try:
        # Get user info using the enhanced agent
        user_info = st.session_state.agent.get_user_info(username)
        
        # Store in session
        st.session_state.username = username
        st.session_state.user_info = user_info
        st.session_state.authenticated = True
        
        # Check if user needs parental setup
        chat_count = check_user_chat_count(username)
        if chat_count == 0:
            st.session_state.needs_parental_setup = True
        else:
            st.session_state.needs_parental_setup = False
            # Load user's previous chat history automatically on login
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
    # Check if user needs parental setup
    if st.session_state.get('needs_parental_setup', False):
        st.title("👨‍👩‍👧‍👦 Parental Setup")
        st.markdown("### Welcome to Lumo! Let's set up your profile.")
        
        with st.form("parental_setup"):
            child_name = st.text_input("Child's Name:", placeholder="Enter your child's name")
            date_of_birth = st.date_input("Date of Birth:", min_value=datetime(2010, 1, 1), max_value=datetime.now())
            interests = st.text_area("Interests:", placeholder="What does your child like? (e.g., dinosaurs, space, art)")
            topics_to_avoid = st.text_area("Topics to Avoid:", placeholder="What topics should we avoid? (e.g., scary movies, certain games)")
            
            submit = st.form_submit_button("Complete Setup")
            
            if submit:
                if child_name.strip():
                    # Calculate age from date of birth
                    today = datetime.now().date()
                    age = (today - date_of_birth).days // 365
                    
                    # Convert date_of_birth to datetime for MongoDB
                    date_of_birth_datetime = datetime.combine(date_of_birth, datetime.min.time())
                    
                    # Format interests (limit to 3)
                    interests_list = [i.strip() for i in interests.split(",") if i.strip()][:3]
                    interests_text = ", ".join(interests_list) if interests_list else "fun things"
                    
                    # Create profile object
                    profile = {
                        "child_name": child_name,
                        "date_of_birth": date_of_birth_datetime,  # Using datetime instead of date
                        "age": age,
                        "interests": interests_text,  # Store formatted interests text
                        "topics_to_avoid": topics_to_avoid,
                        "parental_setup_complete": True,
                        "created_at": datetime.utcnow(),
                        "updated_at": datetime.utcnow(),
                        "interaction_count": 0,
                        "current_mode": "general",
                        "current_emotion": "neutral",
                        "email": f"{st.session_state.username}@lumo.ai",
                        "user_timezone": "UTC"
                    }
                    
                    # Save to MongoDB
                    try:
                        mongo_client = MongoClient(MONGODB_URI)
                        db = mongo_client[DATABASE_NAME]
                        
                        # Update user document with profile
                        db.users.update_one(
                            {"username": st.session_state.username},
                            {
                                "$set": {
                                    "profile": profile,
                                    "interaction_count": 0,
                                    "created_at": datetime.utcnow(),
                                    "updated_at": datetime.utcnow()
                                }
                            },
                            upsert=True
                        )
                        
                        # Print debug info
                        print(f"Debug - Profile being saved: {profile}")
                        print(f"Debug - Interests being saved: {profile.get('interests')}")
                        
                        # Update session state
                        st.session_state.needs_parental_setup = False
                        st.session_state.user_info = profile
                        
                        # Get first message from Lumo
                        initial_message = st.session_state.agent.process_message("", st.session_state.username)
                        st.session_state.messages = [{"role": "assistant", "content": initial_message}]
                        
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error saving setup: {e}")
                else:
                    st.warning("Please enter your child's name")
        st.stop()
    
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
                created_at = st.session_state.user_info.get('created_at')
                if isinstance(created_at, datetime):
                    st.write(f"**Created:** {created_at.strftime('%Y-%m-%d')}")
                else:
                    st.write(f"**Created:** {created_at}")
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

# Chat interface
st.markdown("### �� Chat with Lumo")

# Display chat messages
for message in st.session_state.messages:
    if message["role"] == "user":
        st.chat_message("user").write(message["content"])
    else:
        st.chat_message("assistant").write(message["content"])

# Chat input
if prompt := st.chat_input("Type your message here..."):
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    
    # Get AI response
    try:
        response = st.session_state.agent.process_message(prompt, st.session_state.username)
        
        # Add AI response to chat
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.chat_message("assistant").write(response)
        
    except Exception as e:
        st.error(f"Error: {str(e)}")
        print(f"Error in chat: {e}") 