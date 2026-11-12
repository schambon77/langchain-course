import os

from dotenv import load_dotenv

from langchain_anthropic import ChatAnthropic

load_dotenv()


def main() -> None:
    print("Hello from langchain-course!")


if __name__ == "__main__":
    main()
