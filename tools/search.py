"""Web search tool using the Tavily API."""

import os
import sys
from tavily import TavilyClient


def _clean(text: str) -> str:
    """Replace characters that can't be encoded on the current terminal."""
    encoding = sys.stdout.encoding or "utf-8"
    return text.encode(encoding, errors="replace").decode(encoding)


def web_search(query: str) -> str:
    """Search the web using Tavily and return results as a formatted string.

    Args:
        query: The search query string.

    Returns:
        A string containing the search results, or an error message if
        the API key is missing or the request fails.
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return "Error: TAVILY_API_KEY environment variable is not set."

    client = TavilyClient(api_key=api_key)

    try:
        response = client.search(query=query, search_depth="basic")
        results = response.get("results", [])

        if not results:
            return f"No results found for: {query}"

        parts = [f"Search results for: {query}\n"]
        for i, r in enumerate(results, 1):
            title = r.get("title", "No title")
            url = r.get("url", "")
            content = r.get("content", "No content")
            parts.append(f"{i}. {title}")
            parts.append(f"   URL: {url}")
            parts.append(f"   {content}\n")

        return _clean("\n".join(parts).strip())

    except Exception as e:
        return f"Error performing search: {e}"
