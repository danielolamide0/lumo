import streamlit as st
import uuid
from ai_toy_agent import LumoAgent, DEFAULT_AI_TOY_SYSTEM_PROMPT

st.set_page_config(page_title="Lumo AI Playground", layout="wide", initial_sidebar_state="collapsed")

if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = str(uuid.uuid4())

if "current_system_prompt" not in st.session_state:
    st.session_state.current_system_prompt = DEFAULT_AI_TOY_SYSTEM_PROMPT

if "agent" not in st.session_state:
    st.session_state.agent = LumoAgent(initial_system_prompt=st.session_state.current_system_prompt)
    if not st.session_state.agent.llm:
        st.error("Fatal Error: Language Model (LLM) could not be initialized. Please check your .env file and Google Gemini API Key.")
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

st.title("🧸 Lumo AI Toy - Interactive Playground")
st.markdown("Chat with Lumo and experiment by changing its personality (system prompt)!")

with st.expander("⚙️ Configure Lumo's System Prompt", expanded=False):
    edited_prompt = st.text_area(
        "Current System Prompt for Lumo:",
        value=st.session_state.current_system_prompt,
        height=300,
        key="system_prompt_editor_area"
    )

    if st.button("✨ Apply New System Prompt", key="apply_prompt_button"):
        st.session_state.current_system_prompt = edited_prompt
        st.session_state.agent.update_system_prompt(edited_prompt)
        st.success("System prompt updated! Lumo will use this for the next interaction.")
        st.info("You might want to 'Clear Chat & Reset Lumo' from the sidebar for a fresh start with the new prompt.")

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
    st.header("Controls")
    if st.button("🧹 Clear Chat & Reset Lumo"):
        st.session_state.messages = []
        st.session_state.conversation_id = str(uuid.uuid4())
        st.session_state.agent = LumoAgent(initial_system_prompt=st.session_state.current_system_prompt)
        if not st.session_state.agent.llm:
            st.error("Failed to re-initialize the Language Model after reset.")
        else:
            with st.spinner("Lumo is waking up after reset..."):
                ai_greeting = st.session_state.agent.invoke_agent(
                    "Hello!",
                    st.session_state.conversation_id
                )
            st.session_state.messages.append({"role": "assistant", "content": ai_greeting})
            st.success("Chat cleared and Lumo reset with the current system prompt.")
        st.experimental_rerun()

    st.markdown("---")
    st.markdown("Modify the system prompt in the main panel to change Lumo's behavior.")