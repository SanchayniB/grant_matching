from google.cloud import firestore
from langchain_google_firestore import FirestoreChatMessageHistory
import os
from dotenv import load_dotenv
from pathlib import Path
from langchain_mistralai.chat_models import ChatMistralAI


###### Setting up Mistral
dotenv_path = Path('/Users/sanchaynibagade/Documents/github/grant_matching/env/mistral.env')
load_dotenv(dotenv_path=dotenv_path)
MY_KEY = os.getenv('MISTRAL_KEY')
os.environ["MISTRAL_API_KEY"] = MY_KEY

llm_mistral = ChatMistralAI(
    model="mistral-small-latest",
    temperature=0
)

###### Setup Firebase Firestore

PROJECT_ID = "langchain-6fab8"
SESSION_ID = "user_session_ron"  # This could be a username or a unique ID
COLLECTION_NAME = "chat_history"
SERVICE_ACCOUNT_NAME = "llmservice"


# Initialize Firestore Client
print("Initializing Firestore Client...")
client = firestore.Client(project=PROJECT_ID)

# Initialize Firestore Chat Message History
print("Initializing Firestore Chat Message History...")
chat_history = FirestoreChatMessageHistory(
    session_id=SESSION_ID,
    collection=COLLECTION_NAME,
    client=client,
)
print("Chat History Initialized.")
print("Current Chat History:", chat_history.messages)

print("Start chatting with the AI. Type 'exit' to quit.")

while True:
    human_input = input("User: ")
    if human_input.lower() == "exit":
        break

    chat_history.add_user_message(human_input)

    ai_response = llm_mistral.invoke(chat_history.messages)
    chat_history.add_ai_message(ai_response.content)

    print(f"AI: {ai_response.content}")