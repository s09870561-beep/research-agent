"""Quick test for the web_search tool."""

from dotenv import load_dotenv
from tools.search import web_search

load_dotenv()

result = web_search("latest AI agent frameworks 2026")
print(result)
