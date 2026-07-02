import os
import sys
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# Path to the config file
CONFIG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "Config", "config.properties"))

def load_config(path):
    config = {}
    if not os.path.exists(path):
        print(f"[WARNING] Config file not found at: {path}")
        return config
    
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip()
                # Clean up potential double-equal sign issues from LANGCHAIN_API_KEY==...
                if val.startswith("="):
                    val = val[1:].strip()
                config[key] = val
    return config

# Load properties and set environment variables
config = load_config(CONFIG_PATH)
for k, v in config.items():
    os.environ[k] = v

# Disable LangSmith tracing to prevent 403 Forbidden errors if the LangChain API key is invalid/expired
os.environ["LANGCHAIN_TRACING_V2"] = "false"

# Ensure API key is set
if "GEMINI_API_KEY" not in os.environ:
    print("[ERROR] GEMINI_API_KEY not found in config.properties or environment.")
    sys.exit(1)

# Initialize the Gemini LLM
# Using gemini-1.5-flash as default stable model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

# Define prompt template
prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="You are a helpful, friendly assistant. Answer questions clearly and concisely."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

# Chain
chain = prompt | llm

# Memory / History storage
history_store = {}

def get_session_history(session_id: str):
    if session_id not in history_store:
        history_store[session_id] = InMemoryChatMessageHistory()
    return history_store[session_id]

# Wrapped chain with history management
chatbot_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

def run_cli():
    print("==================================================")
    print("      GEMINI CHATBOT (LANGCHAIN) INITIALIZED      ")
    print("==================================================")
    print("Type your message and press Enter. Type 'exit' to quit.")
    print("--------------------------------------------------")
    
    session_id = "default_session"
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break
                
            response = chatbot_with_history.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": session_id}}
            )
            print(f"\nAssistant: {response.content}")
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")

if __name__ == "__main__":
    run_cli()
