from dotenv import load_dotenv
from langchain.messages import HumanMessage, SystemMessage, ToolMessage
from langchain.tools import tool
from langsmith import traceable
from langchain.chat_models import init_chat_model

MAX_ITERATIONS = 10

load_dotenv()

prices = {
    "laptop": 999,
    "headphones": 299,
    "keyboard": 99
}

tiers = {
    "bronze": 5,
    "silver": 10,
    "gold": 15
}

@tool
def get_product_price(product_name: str) -> float:
    """
    Get the price of a product from the catalog.
    """
    print(f"Getting price for product: {product_name}")
    return prices.get(product_name, 0)

@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """
    Apply a discount to a price.
    """
    print(f"Applying discount {discount_tier} to price: {price}")
    discount = tiers.get(discount_tier, 0)
    return round(price * (1 - discount / 100), 2)

@traceable(name="Langchain Agent Loop")
def run_agent(question: str):
    #tools = [get_product_price, apply_discount]

    tools = [get_product_price, apply_discount]
    tools_dict = {tool.name: tool for tool in tools}

    llm = init_chat_model(model="ollama:gpt-oss:20b", temperature=0)
    llm_with_tools = llm.bind_tools(tools)    

    print(f"Question: {question}")
    print(f"=" * 60)

    messages = [
        SystemMessage(content="You are a helpful assistant that can provide product " \
                              "prices and apply discounts."),
        HumanMessage(content=question)
    ]

    for iteration in range(MAX_ITERATIONS):
        print(f"Iteration {iteration + 1}")

        ai_message = llm_with_tools.invoke(messages)

        tool_calls = ai_message.tool_calls

        if not tool_calls:
            print(f"Final answer: {ai_message.content}")
            return ai_message.content

        tool_to_call = tool_calls[0]
        tool_to_call_name = tool_to_call.get("name")
        tool_to_call_args = tool_to_call.get("args", {})
        tool_to_call_id = tool_to_call.get("id")

        print(f"Tool to call: {tool_to_call_name} with args: {tool_to_call_args}")

        tool_to_use = tools_dict.get(tool_to_call_name)

        if tool_to_use is None:
            print(f"Tool {tool_to_call_name} not found.")
            return "Error: Tool not found."

        observation = tool_to_use.invoke(tool_to_call_args)
        print(f"Observation: {observation}")

        messages.append(ai_message)
        messages.append(ToolMessage(content=str(observation), 
                                    tool_name=tool_to_call_name, 
                                    tool_call_id=tool_to_call_id))

    print("Max iterations reached without a final answer.")
    return None

if __name__ == "__main__":
    print("Hello Langchain Agent (.bind_tools)")
    question = "What is the price of a laptop with a silver discount?"
    result = run_agent(question)



