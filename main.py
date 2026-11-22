from dotenv import load_dotenv
from typing import List
from pydantic import BaseModel, Field
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic
from langchain_tavily import TavilySearch

load_dotenv()

class Source(BaseModel):
    """Source for search agent"""
    url: str = Field(description="URL of the source")

class AgentResponse(BaseModel):
    """Response from the search agent"""
    answer: str = Field(description="Answer from the agent")
    sources: List[Source] = Field(description="List of sources found by the agent")

llm = ChatAnthropic(model="claude-opus-4-8")
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)

def main() -> None:
    print("Hello from langchain-course!")
    result = agent.invoke({"messages":HumanMessage(content="Search for 3 job postings for an " \
                                                           "AI engineer using langchain in " \
                                                           "the bay area on LinkedIn and list" \
                                                           " their details")})
    print(result)

if __name__ == "__main__":
    main()

