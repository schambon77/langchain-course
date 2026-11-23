from dotenv import load_dotenv
from langsmith import traceable
import ollama


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

@traceable(name="Get Product Price Tool", run_type="tool")
def get_product_price(product_name: str) -> float:
    """
    Get the price of a product from the catalog.
    """
    print(f"Getting price for product: {product_name}")
    return prices.get(product_name, 0)

@traceable(name="Apply Discount Tool", run_type="tool")
def apply_discount(price: float, discount_tier: str) -> float:
    """
    Apply a discount to a price.
    """
    print(f"Applying discount {discount_tier} to price: {price}")
    discount = tiers.get(discount_tier, 0)
    return round(price * (1 - discount / 100), 2)

tools_for_llm = [
    {
        "type": "function",
        "function": {
            "name": "get_product_price",
            "description": "Look up the price of a product in the catalog.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_name": {
                        "type": "string",
                        "description": "The product name, e.g. 'laptop', 'headphones', 'keyboard'",
                    },
                },
                "required": ["product_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_discount",
            "description": "Apply a discount tier to a price and return the final price. Available tiers: bronze, silver, gold.",
            "parameters": {
                "type": "object",
                "properties": {
                    "price": {"type": "number", "description": "The original price"},
                    "discount_tier": {
                        "type": "string",
                        "description": "The discount tier: 'bronze', 'silver', or 'gold'",
                    },
                },
                "required": ["price", "discount_tier"],
            },
        },
    },
]

@traceable(name="Ollama Chat", run_type="llm")
def ollama_chat_traced(messages):
    return ollama.chat(messages=messages, model="gpt-oss:20b", tools=tools_for_llm)

@traceable(name="Ollama Agent Loop")
def run_agent(question: str):

    tools_dict = {
        "get_product_price": get_product_price,
        "apply_discount": apply_discount
    }

    print(f"Question: {question}")
    print(f"=" * 60)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful shopping assistant. "
                "You have access to a product catalog tool "
                "and a discount tool.\n\n"
                "STRICT RULES — you must follow these exactly:\n"
                "1. NEVER guess or assume any product price. "
                "You MUST call get_product_price first to get the real price.\n"
                "2. Only call apply_discount AFTER you have received "
                "a price from get_product_price. Pass the exact price "
                "returned by get_product_price — do NOT pass a made-up number.\n"
                "3. NEVER calculate discounts yourself using math. "
                "Always use the apply_discount tool.\n"
                "4. If the user does not specify a discount tier, "
                "ask them which tier to use — do NOT assume one."
            ),
        },
        {"role": "user", "content": question},
    ]

    for iteration in range(MAX_ITERATIONS):
        print(f"Iteration {iteration + 1}")

        ai_message = ollama_chat_traced(messages=messages).message

        tool_calls = ai_message.tool_calls

        if not tool_calls:
            print(f"Final answer: {ai_message.content}")
            return ai_message.content

        tool_to_call = tool_calls[0]
        tool_to_call_name = tool_to_call.function.name
        tool_to_call_args = tool_to_call.function.arguments

        print(f"Tool to call: {tool_to_call_name} with args: {tool_to_call_args}")

        tool_to_use = tools_dict.get(tool_to_call_name)

        if tool_to_use is None:
            print(f"Tool {tool_to_call_name} not found.")
            return "Error: Tool not found."

        observation = tool_to_use(**tool_to_call_args)
        print(f"Observation: {observation}")

        messages.append(ai_message)
        messages.append({
            "role": "tool",
            "content": str(observation)})
        
    print("Max iterations reached without a final answer.")
    return None

if __name__ == "__main__":
    print("Hello Langchain Agent (.bind_tools)")
    question = "What is the price of a laptop with a silver discount?"
    result = run_agent(question)



