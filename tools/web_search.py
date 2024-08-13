import os
from typing import Dict, List
from langchain.utilities import SerpAPIWrapper


class WebSearchTool:
    def __init__(self):
        self.search = SerpAPIWrapper()

    def search(self, query: str) -> str:
        try:
            results = self.search.run(query)
            formatted_results = self._format_results(results)
            return formatted_results
        except Exception as e:
            return f"An error occurred during the web search: {str(e)}"

    def _format_results(self, results: List[Dict]) -> str:
        formatted_results = []
        for result in results:
            title = result.get('title', 'No title')
            snippet = result.get('snippet', 'No snippet available')
            link = result.get('link', 'No link available')
            formatted_result = f"Title: {title}\nSnippet: {snippet}\nLink: {link}\n"
            formatted_results.append(formatted_result)
        
        return "\n".join(formatted_results)