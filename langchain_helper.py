from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
import os

os.environ['OPENAI_API_KEY'] = ""

llm = OpenAI(temperature = 0.6)

def generate_restaurant_name_and_items(cuisine):

    # chain 1 : restaurant name

    prompt_name = PromptTemplate(
        input_variables=['cuisine'],
        template="i want to open a restaurant for {cuisine} food.suggest a fency name for this"
    )

    name_chain = LLMChain(llm=llm, prompt=prompt_name, output_key="restaurant_name")

    # Chain 2: Menu Items

    prompt_items = PromptTemplate(
        input_variables=['restaurant_name'],
        template="suggest some menu items for {restaurant_name}. Return it as comma separated list"
    )
    food_items_chain = LLMChain(llm=llm, prompt=prompt_items, output_key="menu_items")

    chain = SequentialChain(
        chains=[name_chain, food_items_chain],
        input_variables=['cuisine'],
        output_variables=['restaurant_name', 'menu_items']
    )

    response = chain({'cuisine': cuisine})

    return response

if __name__ == "__main__":
    print(generate_restaurant_name_and_items("Italian"))
