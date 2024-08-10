from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent
from langchain.prompts import StringPromptTemplate
from langchain import OpenAI, LLMChain
from typing import List, Union


class ResearchAgent:
    def __init__(self):
        self.llm = OpenAI(temperature=0)
        self.tools = [
            Tool(
                name='Web Search',
                func=WebSearchTool().search,
                description="Useful for searching the internet for current information on a topic."
            ),
            Tool(
                name='Wikipedia',
                func=WikipediaTool().lookup,
                description="Useful for getting detailed background information on a topic."
            ),
            Tool(
                name='Summarize',
                func=SummarizeTool().summarize,
                description="Useful for summarizing long pieces of text."
            )
        ]
        self.prompt = CustomPromptTemplate(
            template="You are a research assistant. Your task is to research the following topic: {topic}\n\n"
                     "To do this, you can use the following tools:\n"
                     "{tool_names}\n\n"
                     "Use the following format:\n"
                     "Thought: Consider what you need to do\n"
                     "Action: The action to take, should be one of {tool_names}\n"
                     "Action Input: The input to the action\n"
                     "Observation: The result of the action\n"
                     "... (this Thought/Action/Action Input/Observation can repeat N times)\n"
                     "Thought: I now know the final answer\n"
                     "Final Answer: The final answer to the original input question\n\n"
                     "Begin!\n\n"
                     "Topic: {topic}\n\n"
                     "Thought: {agent_scratchpad}",
            input_variables=["topic", "tool_names", "agent_scratchpad"]
        )
        self.llm_chain = LLMChain(llm=self.llm, prompt=self.prompt)
        self.agent = LLMSingleActionAgent(
            llm_chain=self.llm_chain,
            output_parser=CustomOutputParser(),
            stop=["\nObservation:"],
            allowed_tools=[tool.name for tool in self.tools]
        )
        self.agent_executor = AgentExecutor.from_agent_and_tools(
            agent=self.agent, tools=self.tools, verbose=True
        )

    def research(self, topic: str) -> str:
        return self.agent_executor.run(topic)
