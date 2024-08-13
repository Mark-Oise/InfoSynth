from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent
from langchain.prompts import StringPromptTemplate
from langchain.schema import AgentAction, AgentFinish
from langchain import OpenAI, LLMChain
from tools.web_search import WebSearchTool
from typing import List, Union
import re


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


class CustomPromptTemplate(StringPromptTemplate):
    template:str,
    tools : List[Tool]

    def format(self, **kwargs) --> str:
        intermediate_steps = kwargs.pop('intermediate_steps', [])
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f'\nObservation:{observation}\nThought: '
        kwargs['agent_scratchpad'] = thoughts
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        return self.template.format(**kwargs)


class CustomOutputParser:
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Check for Final Answer
        if 'Final Answer:' in llm_output:
            return AgentFinish(
                return_values={'output': llm_output.split('Final Answer:')[-1].strip()},
                log=llm_output
            )
        
        # Parse Action and Action Input
        action_match = re.search(r'Action:\s*(.*?)(?:\n|$)', llm_output, re.DOTALL)
        action_input_match = re.search(r'Action Input:\s*(.*?)(?:\n|$)', llm_output, re.DOTALL)

        if not action_match or not action_input_match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        
        action = action_match.group(1).strip()
        action_input = action_input_match.group(1).strip()

        return AgentAction(tool=action, tool_input=action_input, log=llm_output)