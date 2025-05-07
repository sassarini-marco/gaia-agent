import time
import warnings
from smolagents import (
    GoogleSearchTool,
    WikipediaSearchTool,
    DuckDuckGoSearchTool
)

class SearchFailedError(Exception):
    """Custom exception for search failures after retries."""
    pass

class RateLimitedSearchTool:
    """Base class for search tools with rate limiting and retries."""

    def __init__(self, min_request_interval=1.0, max_retries=3):
        self.last_request_time = 0
        self.min_request_interval = min_request_interval
        self.max_retries = max_retries

    def _apply_rate_limit(self):
        """Enforce rate limiting between requests."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            time.sleep(sleep_time)
        self.last_request_time = time.time()

    def _handle_retries(self, search_func, *args, **kwargs):
        """Apply retry logic with exponential backoff."""
        for attempt in range(self.max_retries):
            try:
                self._apply_rate_limit()
                result = search_func(*args, **kwargs)
                return result
            except Exception as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    wait_time = (2 ** attempt) * self.min_request_interval
                    time.sleep(wait_time)
                    continue
                error_message = f"Search failed after {self.max_retries} attempts. Last error: {str(last_exception)}"
                warnings.warn(error_message, UserWarning)
                raise SearchFailedError(error_message) from last_exception

        raise SearchFailedError(f"Search failed unexpectedly after {self.max_retries} attempts.")


class EnhancedGoogleSearchTool(GoogleSearchTool, RateLimitedSearchTool):
    """Google Search tool with rate limiting and retries."""

    def __init__(self, provider="serper", min_request_interval=1.0, max_retries=3, **kwargs):
        GoogleSearchTool.__init__(self, provider=provider)
        RateLimitedSearchTool.__init__(self, min_request_interval=min_request_interval, max_retries=max_retries)
        self.name = "google_search_robust" # More descriptive name
        self.description = f"Performs a Google search via {provider} with built-in retries and rate limiting. Use this as your primary search tool."

    def forward(self, query: str, filter_year: int | None = None) -> str:
        """Override forward method to add rate limiting and retries."""
        return self._handle_retries(super().forward, query=query, filter_year=filter_year)

class EnhancedDuckDuckGoSearchTool(DuckDuckGoSearchTool, RateLimitedSearchTool):
    """DuckDuckGo Search tool with rate limiting and retries."""

    def __init__(self, min_request_interval=1.0, max_retries=3, **kwargs):
        DuckDuckGoSearchTool.__init__(self, **kwargs)
        RateLimitedSearchTool.__init__(self, min_request_interval=min_request_interval, max_retries=max_retries)
        self.name = "duckduckgo_search_robust"
        self.description = "Performs a DuckDuckGo search with built-in retries and rate limiting. Use this as a fallback if Google search fails or yields poor results."

    def forward(self, query: str) -> str:
        """Override forward method to add rate limiting and retries."""
        return self._handle_retries(super().forward, query=query)


class EnhancedWikipediaSearchTool(WikipediaSearchTool, RateLimitedSearchTool):
    """Wikipedia Search tool with rate limiting and retries."""

    def __init__(self, min_request_interval=0.5, max_retries=3, **kwargs):
        WikipediaSearchTool.__init__(self, **kwargs)
        RateLimitedSearchTool.__init__(self, min_request_interval=min_request_interval, max_retries=max_retries)
        self.name = "wikipedia_search_robust"
        self.description = "Searches Wikipedia with built-in retries and rate limiting. Use this for encyclopedic information."

    def forward(self, query: str) -> str:
        """Override forward method to add rate limiting and retries."""
        return self._handle_retries(super().forward, query=query)