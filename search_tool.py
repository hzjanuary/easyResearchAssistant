"""Search Tool - Real-Time Web Search for RAG (Retrieval-Augmented Generation)."""
import logging
from typing import List, Dict, Optional
from datetime import datetime

logger = logging.getLogger("search_tool")

# Try to import duckduckgo_search
try:
    from duckduckgo_search import DDGS
    DDGS_AVAILABLE = True
except ImportError:
    DDGS_AVAILABLE = False
    logger.warning("duckduckgo-search not installed. Web search will be disabled.")


def get_web_search(query: str, max_results: int = 3) -> str:
    """
    Perform a web search using DuckDuckGo and return formatted results.
    
    Args:
        query: The search query string
        max_results: Maximum number of results to return (default: 3)
    
    Returns:
        A formatted string containing search results with titles, snippets, and URLs.
        Returns an empty string if search fails or no results found.
    """
    if not DDGS_AVAILABLE:
        logger.warning("DuckDuckGo Search not available")
        return ""
    
    try:
        logger.info(f"Searching web for: {query[:50]}...")
        
        with DDGS() as ddgs:
            results: List[Dict] = list(ddgs.text(query, max_results=max_results))
        
        if not results:
            logger.info("No search results found")
            return ""
        
        # Format results into a readable string
        formatted_results = format_search_results(results)
        logger.info(f"Found {len(results)} search results")
        
        return formatted_results
    
    except Exception as e:
        logger.error(f"Web search failed: {str(e)}")
        return ""


def format_search_results(results: List[Dict]) -> str:
    """
    Format search results into a structured string for LLM consumption.
    
    Args:
        results: List of search result dictionaries from DuckDuckGo
    
    Returns:
        Formatted string with numbered results
    """
    if not results:
        return ""
    
    formatted_parts = []
    current_year = datetime.now().year
    
    for i, result in enumerate(results, 1):
        title = result.get("title", "No title")
        body = result.get("body", result.get("snippet", "No description"))
        url = result.get("href", result.get("link", ""))
        
        # Clean up the body text
        body = body.strip()
        if len(body) > 500:
            body = body[:497] + "..."
        
        formatted_parts.append(
            f"[{i}] {title}\n"
            f"    {body}\n"
            f"    Source: {url}"
        )
    
    header = f"=== Web Search Results (as of {current_year}) ==="
    return header + "\n\n" + "\n\n".join(formatted_parts)


def build_research_prompt(user_query: str, search_results: str) -> str:
    """
    Build a system prompt that includes web search results for RAG.
    
    Args:
        user_query: The original user query
        search_results: Formatted search results string
    
    Returns:
        A system prompt instructing the LLM to use the search results
    """
    current_year = datetime.now().year
    
    if search_results:
        return f"""You are an intelligent research assistant with access to real-time information.

IMPORTANT: Use the following web search results to provide accurate, up-to-date information for the year {current_year}. 
Cite your sources when appropriate by referencing the result numbers [1], [2], etc.

{search_results}

INSTRUCTIONS:
1. Answer the user's question using the search results above when relevant.
2. If the search results contain the answer, prioritize that information.
3. If the search results don't fully answer the question, use your knowledge but mention any limitations.
4. Always provide clear, well-structured responses.
5. If information might be outdated, note that to the user."""
    else:
        return f"""You are an intelligent research assistant.
The current year is {current_year}. Provide accurate, well-researched responses.
If you're unsure about recent events or data, acknowledge the limitation."""


async def async_get_web_search(query: str, max_results: int = 3) -> str:
    """
    Async wrapper for web search (runs sync search in thread pool).
    
    Args:
        query: The search query string
        max_results: Maximum number of results to return
    
    Returns:
        Formatted search results string
    """
    import asyncio
    
    # Run the sync function in a thread pool to avoid blocking
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, get_web_search, query, max_results)
