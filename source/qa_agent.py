import sys
from ollama import chat

def llama3(messages):
    response = chat(model='llama3.2', messages=messages)  # Ensure messages is a list of dicts
    return response['message']["content"]  # Extract content correctly

if __name__ == "__main__":
    # Store conversation history
    messages = [
        {
            "role": "system",
            "content": "You are a concise assistant. Provide answers in 2-3 sentences maximum."
        }
    ]  # Initialize with system message

    print("Welcome to the Ollama Concise Q&A Chat! Type 'exit' to quit.")
    
    while True:
        prompt = input("You: ")
        if prompt.lower() == "exit":
            print("Goodbye!")
            break

        # Append user input correctly
        messages.append({"role": "user", "content": prompt})  

        # Call chat function with correctly structured messages list
        response = llama3(messages)

        # Append model response correctly
        messages.append({"role": "assistant", "content": response})  

        print(f"Ollama: {response}")
