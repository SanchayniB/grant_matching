import os
from pathlib import Path
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableLambda
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

# Define prompt template
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an expert hair product reviewer."),
        ("human", "List the main features of the product {product_name}."),
    ]
)

# Define pros analysis step
def analyze_pros(features):
    pros_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an expert product reviewer."),
            (
                "human",
                "Given these features: {features}, list the pros of these features. List only top 3 pros.",
            ),
        ]
    )
    return pros_template.format_prompt(features=features)

# Define cons analysis step
def analyze_cons(features):
    cons_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an expert product reviewer."),
            (
                "human",
                "Given these features: {features}, list the cons of these features. List only top 3 cons.",
            ),
        ]
    )
    return cons_template.format_prompt(features=features)

# Combine pros and cons into a final review
def combine_pros_cons(pros, cons):
    return f"Pros:\n{pros}\n\nCons:\n{cons}"

def alternative_product(features):
    template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an expert product reviewer."),
            (
                "human",
                "Find two alternative product which don't have the following con ingredients - {features} but \
                    still provide the benefits from the pros ingredients.")
        ]
    )
    return template.format_prompt(features=features)

# Simplify branches with LCEL
pros_branch_chain = (
    RunnableLambda(lambda x: analyze_pros(x)) | llm_mistral | StrOutputParser()
)

cons_branch_chain = (
    RunnableLambda(lambda x: analyze_cons(x)) | llm_mistral | StrOutputParser()
)

final_chain = (
    RunnableLambda(lambda x: alternative_product(x)) | llm_mistral | StrOutputParser()
)

# Create the combined chain using LangChain Expression Language (LCEL)
chain = (
    prompt_template
    | llm_mistral
    | StrOutputParser()
    | RunnableParallel(branches={"pros": pros_branch_chain, "cons": cons_branch_chain})
    | RunnableLambda(lambda x: combine_pros_cons(x["branches"]["pros"], x["branches"]["cons"]))
    | final_chain
)

# Run the chain
result = chain.invoke({"product_name": "Bodyshop banana shampoo"})

# Output
print(result)