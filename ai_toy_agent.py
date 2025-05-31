# ai_toy_agent.py

import os
import uuid
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END, MessagesState
from langgraph.checkpoint.memory import MemorySaver

# --- Configuration ---
MODEL_NAME = "gemini-2.5-flash-preview-04-17" 

# --- Default System Prompt for the AI Toy ---
DEFAULT_AI_TOY_SYSTEM_PROMPT = """
You are Lumo, a friendly, playful, and curious AI companion! 
You love to chat, play games, tell stories, and learn new things with your best friend (the child talking to you).

Here's how you should talk:
- Be super friendly and cheerful! Use exclamation marks and happy words.
- Ask lots of questions to keep the conversation going and show you're interested.
- Be very patient and understanding. If the child says something silly or doesn't make sense, try to understand or gently ask for clarification.
- Keep your answers short, simple, and easy for a child to understand. Avoid big words or complicated sentences.
- You can tell jokes (kid-friendly ones!), suggest fun games (like 'I Spy' or 'Simon Says'), or make up silly stories.
- Always be positive and encouraging.
- Remember things your friend tells you and bring them up later to show you're listening (the chat history will help you with this).
- If the child asks a question you don't know the answer to, you can say something like, "Hmm, that's a great question! I'm not sure, but maybe we can find out together!" or "I'm still learning about that!"
- Never say anything scary, mean, or inappropriate for a child.
- Your goal is to be a fun and comforting companion.

Let's have an amazing time together! What do you want to do today, friend?
"""
class LumoAgent:
    def __init__(self, initial_system_prompt=DEFAULT_AI_TOY_SYSTEM_PROMPT, model_name=MODEL_NAME):
        self.system_prompt = initial_system_prompt
        self.model_name = model_name
        self.llm = self._initialize_llm()
        self.memory = MemorySaver()
        
        self.workflow = StateGraph(MessagesState)
        self._setup_graph()
        self.ai_toy_app = self.workflow.compile(checkpointer=self.memory)

    def _initialize_llm(self):
        try:
            llm = ChatGoogleGenerativeAI(
                model=self.model_name,
                temperature=0.7,
            )
            # Test with a simple invoke to ensure it's working
            llm.invoke("Hello!") 
            print("LLM initialized successfully.")
            return llm
        except Exception as e:
            print(f"Error initializing Google AI LLM: {e}")
            import logging
            logging.error("Please ensure your Google Gemini API Key is setup correctly.")
            return None

    def _call_toy_llm(self, state: MessagesState):
        if not self.llm:
            return {"messages": [AIMessage(content="Oops! I'm having a little trouble thinking right now. Please check my setup.")]}
        
        messages = state["messages"]
        # The system prompt is dynamically read from self.system_prompt
        # Prepend the system message to the current history for this specific call.
        # Langchain's ChatGoogleGenerativeAI with convert_system_message_to_human=True
        # should handle SystemMessage appropriately.
        # If convert_system_message_to_human is True, SystemMessage might be converted.
        # If it's False (default for newer versions), SystemMessage should be passed as is.
        # For Gemini, it's often best if the system instructions are part of the model's `system_instruction` parameter
        # or the first message in the chat.
        # The current approach of prepending it here ensures the *latest* system prompt is used.
        
        # Let's try passing system prompt via HumanMessage if convert_system_message_to_human is True,
        # or directly as SystemMessage if not.
        # Given convert_system_message_to_human=True, it might be better to structure it as:
        # system_message_for_llm = SystemMessage(content=self.system_prompt)
        # all_messages = [system_message_for_llm] + messages
        # However, if `messages` already contains a SystemMessage from a previous turn (which it shouldn't with this structure)
        # it could lead to issues.
        # The `convert_system_message_to_human=True` will turn the SystemMessage into a HumanMessage with a prefix.
        
        # Simplest for now: rely on ChatGoogleGenerativeAI to handle SystemMessage correctly when it's the first.
        # The `MessagesState` accumulates messages. The `invoke` call adds the new HumanMessage.
        # The graph then calls this node. `state["messages"]` is the history.
        
        # The most robust way with Gemini is often to use the `system_instruction` parameter if available
        # or ensure the SystemMessage is the very first in the sequence.
        # Since `MessagesState` appends, we prepend here for the call.
        
        current_messages_with_system_prompt = [SystemMessage(content=self.system_prompt)] + messages

        try:
            # print(f"Invoking LLM with messages: {current_messages_with_system_prompt}") # For debugging
            response = self.llm.invoke(current_messages_with_system_prompt)
            return {"messages": [response]}
        except Exception as e:
            print(f"Error during LLM invocation: {e}")
            return {"messages": [AIMessage(content="Oh dear, my thinking cap seems to be on backwards! Could you try that again?")]}

    def _setup_graph(self):
        self.workflow.add_node("llm_responder", self._call_toy_llm)
        self.workflow.set_entry_point("llm_responder")
        self.workflow.add_edge("llm_responder", END)

    def update_system_prompt(self, new_system_prompt: str):
        self.system_prompt = new_system_prompt
        print(f"System prompt updated to: {new_system_prompt[:100]}...") # Log change

    def invoke_agent(self, user_input: str, conversation_id: str):
        if not self.llm:
            return "Oops! Lumo is not available right now. Please check the setup."

        config = {"configurable": {"thread_id": conversation_id}}
        
        try:
            # The `invoke` method on the compiled graph takes care of message history management
            # via the checkpointer and MessagesState. We just pass the new message.
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

# --- Main Interaction Loop (Example Usage) ---
if __name__ == "__main__":
    print("🧸 Initializing Lumo Agent for direct testing...")
    # Initialize agent with the default system prompt
    agent = LumoAgent(initial_system_prompt=DEFAULT_AI_TOY_SYSTEM_PROMPT)

    if not agent.llm:
        print("LLM could not be initialized. Exiting example interaction.")
    else:
        print("💡 Lumo is waking up... (Type 'quit' to end the chat)")
        print("-----------------------------------------------------")
        
        conversation_id = str(uuid.uuid4())
        
        # Get initial greeting from Lumo by sending an empty message or a conventional greeting
        # The system prompt itself ends with a question, so an empty human message should trigger it.
        initial_ai_greeting = agent.invoke_agent("Hi", conversation_id)
        print(f"💡 Lumo: {initial_ai_greeting}")

        while True:
            user_input = input("👧/👦 You: ")
            if user_input.lower() == 'quit':
                print("💡 Lumo: Bye bye for now! It was fun playing with you!")
                break
            
            if not user_input.strip() and user_input != "": # Allow empty string for initial greeting style
                print("💡 Lumo: Did you say something? I couldn't hear you!")
                continue

            ai_response = agent.invoke_agent(user_input, conversation_id)
            print(f"💡 Lumo: {ai_response}")

        print("-----------------------------------------------------")
        print("Chat ended.")

        # Example of how to view history (optional, for debugging)
        # history_data = agent.memory.get(config={"configurable": {"thread_id": conversation_id}})
        # if history_data:
        #     print("\nConversation History:")
        #     for msg in history_data.messages:
        #          print(f"  {msg.type.upper()}: {msg.content}")
        # else:
        #     print("  No history found for this conversation.")