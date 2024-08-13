from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain.llms import OpenAI


class SummarizeTool:
    def __init__(self):
        self.llm = OpenAI(temperature=0)
        self.chain = load_summarize_chain(self.llm, chain_type=map_reduce)

    def summarize(self, text: str) -> str:
        try:
            docs = [Document(page_context=text)]
            summary = self.chain(docs)
            return summary
        except Exception as e:
            return f'An error occurred during summarization: {str(e)}'