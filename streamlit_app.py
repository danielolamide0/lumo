import streamlit as st
import uuid
from ai_toy_agent import LumoAgent, CORE_IDENTITY_PROMPT, MODE_SPECIFIC_PROMPTS

st.set_page_config(page_title="Lumo AI Playground", layout="wide", initial_sidebar_state="collapsed")

# Initialize session state for all prompt types
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = str(uuid.uuid4())

if "core_identity" not in st.session_state:
    st.session_state.core_identity = CORE_IDENTITY_PROMPT

if "mode_prompts" not in st.session_state:
    st.session_state.mode_prompts = MODE_SPECIFIC_PROMPTS.copy()

if "agent" not in st.session_state:
    st.session_state.agent = LumoAgent(
        core_identity=st.session_state.core_identity,
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

st.title("🧸 Lumo AI Toy - Enhanced Multi-Mode Playground")
st.markdown("Chat with Lumo and customize its core identity and specialized mode behaviors!")

# Enhanced Configuration Section
with st.expander("⚙️ Configure Lumo's Prompts (Best Practice Architecture)", expanded=False):
    
    # Create tabs for different prompt types
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🧠 Core Identity", 
        "💬 Chat Mode", 
        "🎮 Game Mode", 
        "📚 Story Mode", 
        "🎓 Learning Mode",
        "👁️ View Combined"
    ])
    
    with tab1:
        st.subheader("Core Identity (Shared Across All Modes)")
        st.markdown("*This defines Lumo's fundamental personality, safety rules, and communication style.*")
        
        core_identity_edited = st.text_area(
            "Core Identity Prompt:",
            value=st.session_state.core_identity,
            height=300,
            key="core_identity_editor",
            help="This prompt defines Lumo's core personality and will be included in ALL interaction modes."
        )
        
        if st.button("✨ Update Core Identity", key="update_core"):
            st.session_state.core_identity = core_identity_edited
            st.session_state.agent.update_core_identity(core_identity_edited)
            st.success("Core identity updated! This affects all interaction modes.")
    
    with tab2:
        st.subheader("Chat Mode Specialization")
        st.markdown("*Instructions for general conversation and companionship.*")
        
        chat_prompt_edited = st.text_area(
            "Chat Mode Prompt:",
            value=st.session_state.mode_prompts["chat"],
            height=200,
            key="chat_mode_editor"
        )
        
        if st.button("💬 Update Chat Mode", key="update_chat"):
            st.session_state.mode_prompts["chat"] = chat_prompt_edited
            st.session_state.agent.update_mode_prompt("chat", chat_prompt_edited)
            st.success("Chat mode updated!")
    
    with tab3:
        st.subheader("Game Mode Specialization")
        st.markdown("*Instructions for interactive games and playful activities.*")
        
        game_prompt_edited = st.text_area(
            "Game Mode Prompt:",
            value=st.session_state.mode_prompts["game"],
            height=200,
            key="game_mode_editor"
        )
        
        if st.button("🎮 Update Game Mode", key="update_game"):
            st.session_state.mode_prompts["game"] = game_prompt_edited
            st.session_state.agent.update_mode_prompt("game", game_prompt_edited)
            st.success("Game mode updated!")
    
    with tab4:
        st.subheader("Story Mode Specialization")
        st.markdown("*Instructions for storytelling and narrative creation.*")
        
        story_prompt_edited = st.text_area(
            "Story Mode Prompt:",
            value=st.session_state.mode_prompts["story"],
            height=200,
            key="story_mode_editor"
        )
        
        if st.button("📚 Update Story Mode", key="update_story"):
            st.session_state.mode_prompts["story"] = story_prompt_edited
            st.session_state.agent.update_mode_prompt("story", story_prompt_edited)
            st.success("Story mode updated!")
    
    with tab5:
        st.subheader("Learning Mode Specialization")
        st.markdown("*Instructions for educational content and teaching.*")
        
        learning_prompt_edited = st.text_area(
            "Learning Mode Prompt:",
            value=st.session_state.mode_prompts["learning"],
            height=200,
            key="learning_mode_editor"
        )
        
        if st.button("🎓 Update Learning Mode", key="update_learning"):
            st.session_state.mode_prompts["learning"] = learning_prompt_edited
            st.session_state.agent.update_mode_prompt("learning", learning_prompt_edited)
            st.success("Learning mode updated!")
    
    with tab6:
        st.subheader("Combined Prompts Preview")
        st.markdown("*See how the core identity combines with each mode-specific prompt.*")
        
        mode_selector = st.selectbox(
            "Select mode to preview:",
            ["chat", "game", "story", "learning"],
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
    st.subheader("📋 Current Architecture")
    st.markdown("""
    **Best Practice Design:**
    - 🧠 **Core Identity**: Shared personality & safety
    - 💬 **Chat Mode**: General conversation
    - 🎮 **Game Mode**: Interactive games  
    - 📚 **Story Mode**: Storytelling
    - 🎓 **Learning Mode**: Educational content
    
    Each interaction gets **Core + Mode** prompts combined!
    """)
    
    st.markdown("---")
    st.subheader("🧪 Testing Tips")
    st.markdown("""
    **Try these to test AI-powered routing & emotion detection:**
    - "I'm bored, let's do something fun!" → 🎮 Game Mode + Neutral/Bored
    - "I had a bad day, can you tell me a happy story?" → 📚 Story Mode + Sad
    - "I'm so excited! How do rockets work?!" → 🎓 Learning Mode + Excited
    - "Hi! I'm feeling great today!" → 💬 Chat Mode + Happy
    - "I'm confused about math homework" → 🎓 Learning Mode + Confused
    - "I'm tired, tell me a bedtime story" → 📚 Story Mode + Tired
    
    **Lumo now detects both INTENT and EMOTION dynamically!**
    """)
    
    # Show current prompt statistics
    st.markdown("---")
    st.subheader("📊 Prompt Stats")
    core_len = len(st.session_state.core_identity)
    total_modes = len(st.session_state.mode_prompts)
    
    st.metric("Core Identity Length", f"{core_len} chars")
    st.metric("Total Modes", total_modes)
    
    for mode_name, mode_prompt in st.session_state.mode_prompts.items():
        combined_len = len(st.session_state.agent.get_combined_prompt(mode_name))
        st.metric(f"{mode_name.title()} Combined", f"{combined_len} chars")