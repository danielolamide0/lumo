import streamlit as st
import uuid
from ai_toy_agent import LumoAgent, CORE_IDENTITY_PROMPT, CHAT_FOUNDATION_PROMPT, MODE_SPECIFIC_PROMPTS

st.set_page_config(page_title="Lumo AI Playground", layout="wide", initial_sidebar_state="collapsed")

# Initialize session state for all prompt types
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = str(uuid.uuid4())

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
    if st.session_state.agent and st.session_state.agent.llm:
        with st.spinner("Lumo is waking up..."):
            ai_greeting = st.session_state.agent.invoke_agent(
                "Hello!",
                st.session_state.conversation_id
            )
        st.session_state.messages.append({"role": "assistant", "content": ai_greeting})

st.title("🧸 Lumo AI Toy Playground")
st.markdown("Chat with Lumo and customize its core identity and specialized mode behaviors!")

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

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What do you want to say to Lumo?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner("Lumo is thinking..."):
            ai_response_content = st.session_state.agent.invoke_agent(
                prompt,
                st.session_state.conversation_id
            )
            message_placeholder.markdown(ai_response_content)
    
    st.session_state.messages.append({"role": "assistant", "content": ai_response_content})

with st.sidebar:
    st.header("🎛️ Controls")
    
    if st.button("🧹 Clear Chat & Reset Lumo"):
        st.session_state.messages = []
        st.session_state.conversation_id = str(uuid.uuid4())
        st.session_state.agent = LumoAgent(
            core_identity=st.session_state.core_identity,
            chat=st.session_state.chat,
            mode_prompts=st.session_state.mode_prompts
        )
        if not st.session_state.agent.llm:
            st.error("Failed to re-initialize the Language Model after reset.")
        else:
            with st.spinner("Lumo is waking up after reset..."):
                ai_greeting = st.session_state.agent.invoke_agent(
                    "Hello!",
                    st.session_state.conversation_id
                )
            st.session_state.messages.append({"role": "assistant", "content": ai_greeting})
            st.success("Chat cleared and Lumo reset with current prompts.")
        st.experimental_rerun()

    st.markdown("---")
    st.subheader("📋 Simple & Correct Architecture")
    st.markdown("""
    **Every mode gets the same foundation:**
    - 🧠 **Core Identity**: Personality & safety
    - 💬 **Chat**: Conversational abilities (SHARED)
    - 🎯 **+ Specialization**: Game/Story/Learning focus
    
    **No separate chat mode - chat IS the foundation!**
    """)
    
    st.markdown("---")
    st.subheader("🧪 Test It")
    st.markdown("""
    **All modes are conversational + specialized:**
    - "I'm bored!" → 🎮 Gaming conversation
    - "Tell me about space" → 🎓 Educational conversation  
    - "I'm sad" → 💬 Supportive conversation (general)
    - "Story about dragons" → 📚 Storytelling conversation
    
    **Every response = Core + Chat + Specialization!**
    """)
    
    # Show current prompt statistics
    st.markdown("---")
    st.subheader("📊 Architecture Stats")
    core_len = len(st.session_state.core_identity)
    chat_len = len(st.session_state.chat)
    total_modes = len(st.session_state.mode_prompts)
    
    st.metric("Core Identity", f"{core_len} chars")
    st.metric("Chat (Shared)", f"{chat_len} chars")
    st.metric("Specialized Modes", total_modes)
    
    for mode_name, mode_prompt in st.session_state.mode_prompts.items():
        combined_len = len(st.session_state.agent.get_combined_prompt(mode_name))
        st.metric(f"{mode_name.title()} Total", f"{combined_len} chars")
