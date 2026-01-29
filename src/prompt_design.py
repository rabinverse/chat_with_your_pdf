from langchain_core.prompts import PromptTemplate, ChatPromptTemplate


CUSTOM_PROMPT_TEMPLATE = """
use the given context to give the answer to the user Question.
if answer is not in the context don't make up the answer just say i don't know answer is not in the given context
dont provide answer out of the given context
Context:{context}
Question:{question} 
"""


def return_custom_prompt_template():
    prompt = PromptTemplate(
        template=CUSTOM_PROMPT_TEMPLATE, input_variables=["context", "question"]
    )
    return prompt
