import wikipedia
from typing import Optional


class WikipediaTool:
    def lookup(self, query: str) -> str:
        try:
            page = wikipedia.page(query)
            return self._format_result(page)
        except wikipedia.exceptions.DisambiguationError as e:
            return f"Multiple results found. Please be more specific. Options include: {', '.join(e.options[:5])}"
        except wikipedia.exceptions.PageError:
            return f"No Wikipedia page found for '{query}'"'
        except Exception as e:
            return f"An error occurred during Wikipedia lookup: {str(e)}"

    def _format_result(self, page: wikipedia.WikipediaPage) -> str:
        summary = page.summary[:500] + "..." if len(page.summary) > 500 else page.summary
        return f"Title: {page.title}\n\nSummary: {summary}\n\nURL: {page.url}"