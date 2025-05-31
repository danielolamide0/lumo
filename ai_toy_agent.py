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
MODEL_NAME = "gemini-pro"

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
                google_api_key=os.getenv("GOOGLE_API_KEY")
            )
            llm.invoke("Hello!") 
            print("LLM initialized successfully.")
            return llm
        except Exception as e:
            print(f"Error initializing Google AI LLM: {e}")
            print("Please ensure your Google API Key is setup correctly.")
            return None

    def _call_toy_llm(self, state: MessagesState):
        if not self.llm:
            return {"messages": [AIMessage(content="Oops! I'm having a little trouble thinking right now. Please check my setup.")]}
        
        messages = state["messages"]
        current_messages_with_system_prompt = [SystemMessage(content=self.system_prompt)] + messages

        try:
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
        print(f"System prompt updated to: {new_system_prompt[:100]}...")

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
    agent = LumoAgent(initial_system_prompt=DEFAULT_AI_TOY_SYSTEM_PROMPT)

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