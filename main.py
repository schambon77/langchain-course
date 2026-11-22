from dotenv import load_dotenv
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic
from tavily import TavilyClient

load_dotenv()

tavily = TavilyClient()

@tool
def search(query: str) -> str:
    """
    Search a query and return the results as a string.
    Args:
        query (str): The search query.
    Returns:
        str: The search results.
    """
    print(f"Searching for: {query}")
    return tavily.search(query=query)

llm = ChatAnthropic(model="claude-opus-4-8")
tools = [search]
agent = create_agent(model=llm, tools=tools)


def main() -> None:
    print("Hello from langchain-course!")
    result = agent.invoke({"messages":HumanMessage(content="Search for 3 job postings for an " \
                                                           "AI engineer using langchain in " \
                                                           "the bay area on LinkedIn and list" \
                                                           " their details")})
    print(result)

if __name__ == "__main__":
    main()

