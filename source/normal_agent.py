from ollama import chat
from ollama import ChatResponse
import sys

def llama3(prompt, messages=[]):
    if len(messages) == 0:
        messages = [{"role": "user", "content": prompt}]
    
    response: ChatResponse = chat(model='llama3.2', messages=messages)
    return response['message']['content']

if __name__ == "__main__":
    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:])  # Join arguments into a single prompt
    else:
        prompt = input("Enter your prompt: ")  # Interactive mode
    
    res = llama3(prompt)
    print(f'Response from Ollama: {res}')
