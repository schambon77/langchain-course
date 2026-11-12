from dotenv import load_dotenv

from langchain_anthropic import ChatAnthropic

from langchain.prompts import PromptTemplate

load_dotenv()


def main() -> None:
    print("Hello from langchain-course!")

    information = """ 
    Yann André Le Cun[3] (/ləˈkʌn/ lə-KUN; French: [ləkœ̃];[4] usually spelled LeCun;[4] born 8 July 1960) is a French-American computer scientist working in the fields of artificial intelligence, machine learning, computer vision, robotics and image compression.[1][5] He is the Jacob T. Schwartz Professor of Computer Science at the Courant Institute of Mathematical Sciences at New York University (NYU). He served as Chief AI Scientist at Meta Platforms before co-founding Advanced Machine Intelligence Labs in December 2025.[6][7][8]
    LeCun completed a computer science PhD at the Université Pierre et Marie Curie in 1987, and was briefly a postdoctoral researcher under Geoffrey Hinton at the University of Toronto. From 1988, LeCun led the development of the LeNet convolutional neural networks for optical character recognition at Bell Labs.[5][9] He is one of the main creators of the DjVu image compression technology, alongside Léon Bottou and Patrick Haffner. In 2003, LeCun joined NYU, where he worked on energy-based models, computer vision, and DARPA autonomous navigation projects.
    In 2013, LeCun joined Facebook Artificial Intelligence Research. In 2018, LeCun, Yoshua Bengio, and Geoffrey Hinton received the Turing Award from the Association for Computing Machinery for their work on deep learning.[10] LeCun, Bengio, and Hinton are sometimes referred to as the "Godfathers of AI" and "Godfathers of Deep Learning".
    """
    summary_template = """
    given the information {information} about a person I want you to create:
    1. A short summary
    2. two interesting facts about them
    """
    summary_prompt_template = PromptTemplate(
        input_variables=["information"],
        template=summary_template,
    )

    llm = ChatAnthropic(model="claude-opus-4-8", max_tokens=1000)

    chain = summary_prompt_template | llm

    response = chain.invoke({"information": information})
    print(response.content)

if __name__ == "__main__":
    main()
