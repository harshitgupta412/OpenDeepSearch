from typing import Optional, Literal
from smolagents import Tool
from opendeepsearch.ods_agent import OpenDeepSearchAgent
from lotus.web_search import WebSearchCorpus
import datetime
from lotus.models import LM

class OpenDeepSearchTool(Tool):
    name = "web_search"
    description = """
    Performs web search based on your query (think a Google search) then returns the final answer that is processed by an llm. Always use this tool to get facts."""
    inputs = {
        "query": {
            "type": "string",
            "description": "The search query to perform",
        },
    }
    output_type = "string"

    def __init__(
        self,
        model_name: str | LM | None = None,
        reranker: str = "infinity",
        search_provider: Literal["serper", "searxng", "lotus"] = "serper",
        serper_api_key: Optional[str] = None,
        searxng_instance_url: Optional[str] = None,
        searxng_api_key: Optional[str] = None,
        lotus_corpus: list[WebSearchCorpus] = [WebSearchCorpus.GOOGLE, WebSearchCorpus.GOOGLE_SCHOLAR, WebSearchCorpus.BING, WebSearchCorpus.TAVILY],
        lotus_sort_by_date: bool = False,
        lotus_end_date: Optional[datetime.date] = None,
        lotus_multiplier: int = 1,
        max_sources: int = 2,
    ):
        super().__init__()
        self.search_model_name = model_name  # LiteLLM model name
        self.reranker = reranker
        self.search_provider = search_provider
        self.serper_api_key = serper_api_key
        self.searxng_instance_url = searxng_instance_url
        self.searxng_api_key = searxng_api_key
        self.lotus_corpus = lotus_corpus
        self.lotus_sort_by_date = lotus_sort_by_date
        self.lotus_end_date = lotus_end_date
        self.lotus_multiplier = lotus_multiplier
        self.max_sources = max_sources
        self.sources = []
    
    def forward(self, query: str):
        answer, sources = self.search_tool.ask_sync(query, max_sources=self.max_sources, pro_mode=True)
        self.sources += sources["organic"]
        answer += "\n\nSources: " + "\n".join([f"[{source['title']}]({source['link']})" for source in sources["organic"]])
        return answer

    def setup(self):
        self.search_tool = OpenDeepSearchAgent(
            self.search_model_name,
            reranker=self.reranker,
            search_provider=self.search_provider,
            serper_api_key=self.serper_api_key,
            searxng_instance_url=self.searxng_instance_url,
            searxng_api_key=self.searxng_api_key,
            lotus_corpus=self.lotus_corpus,
            lotus_sort_by_date=self.lotus_sort_by_date,
            lotus_end_date=self.lotus_end_date,
            lotus_multiplier=self.lotus_multiplier
        )
