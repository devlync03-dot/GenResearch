import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")


class Settings:
    """
    All configuration lives here. No other config file should be needed.
    Environment variables override defaults.
    """

    # App
    APP_NAME: str = os.getenv("APP_NAME", "GenResearch")
    APP_VERSION: str = os.getenv("APP_VERSION", "0.3.0")
    DEBUG: bool = os.getenv("DEBUG", "False").lower() in ("true", "1", "yes")
    CORS_ORIGINS: list[str] = [
        origin.strip().rstrip("/")
        for origin in os.getenv(
            "CORS_ORIGINS",
            "http://localhost:3000,http://localhost:3001,http://127.0.0.1:3000,http://127.0.0.1:3001",
        ).split(",")
        if origin.strip()
    ]
    # Ollama is used only for document and query embeddings.
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_CONNECT_TIMEOUT: float = float(os.getenv("OLLAMA_CONNECT_TIMEOUT", "5"))
    OLLAMA_REQUEST_TIMEOUT: float = float(os.getenv("OLLAMA_REQUEST_TIMEOUT", "600"))
    OLLAMA_KEEP_ALIVE: str = os.getenv("OLLAMA_KEEP_ALIVE", "30m")
    OLLAMA_EMBED_MODEL: str = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
    OLLAMA_EMBED_PREFIXES: bool = os.getenv(
        "OLLAMA_EMBED_PREFIXES", "false"
    ).lower() in ("true", "1", "yes")
    MODAL_PROXY_TOKEN_ID: str = os.getenv("MODAL_PROXY_TOKEN_ID", "").strip()
    MODAL_PROXY_TOKEN_SECRET: str = os.getenv("MODAL_PROXY_TOKEN_SECRET", "").strip()
    MODAL_BASE_URL: str = os.getenv(
        "MODAL_BASE_URL",
        "https://abbasi-kaleem2002--ep-deepseek-v4-1-flash-1017-server.us-west.modal.direct/v1",
    ).rstrip("/")
    MODAL_MODEL: str = os.getenv(
        "MODAL_MODEL", "deepseek-ai/DeepSeek-V4.1-Flash-1017"
    )
    MODAL_REASONING_EFFORT: str = os.getenv("MODAL_REASONING_EFFORT", "none")
    MODAL_SUMMARIZATION_REASONING_EFFORT: str = os.getenv(
        "MODAL_SUMMARIZATION_REASONING_EFFORT", "none"
    )
    MODAL_LITERATURE_REVIEW_REASONING_EFFORT: str = os.getenv(
        "MODAL_LITERATURE_REVIEW_REASONING_EFFORT", "none"
    )
    MODAL_TOP_P: float = float(os.getenv("MODAL_TOP_P", "0.9"))
    MODAL_MAX_TOKENS: int = int(os.getenv("MODAL_MAX_TOKENS", "16384"))
    MODAL_CONTEXT: int = int(os.getenv("MODAL_CONTEXT", "32768"))
    MODAL_MAX_CONCURRENT: int = int(os.getenv("MODAL_MAX_CONCURRENT", "2"))
    MODAL_CONNECT_TIMEOUT: float = float(os.getenv("MODAL_CONNECT_TIMEOUT", "10"))
    MODAL_READ_TIMEOUT: float = float(os.getenv("MODAL_READ_TIMEOUT", "180"))

    # Document structure and chat routing
    DOC_STRUCTURE_ON_UPLOAD: bool = os.getenv("DOC_STRUCTURE_ON_UPLOAD", "True").lower() in ("true", "1", "yes")
    CHAT_INTENT_ROUTING: bool = os.getenv("CHAT_INTENT_ROUTING", "True").lower() in ("true", "1", "yes")
    DOC_STRUCTURE_VERSION: str = os.getenv("DOC_STRUCTURE_VERSION", "1")
    CHAT_MAP_MAX_CONCURRENCY: int = int(os.getenv("CHAT_MAP_MAX_CONCURRENCY", "2"))
    CHAT_MAP_WINDOW_CHARS: int = int(
        os.getenv("CHAT_MAP_WINDOW_CHARS", str(max(2000, MODAL_CONTEXT // 3)))
    )

    # Chat retrieval quality
    CHAT_TOP_K: int = int(os.getenv("CHAT_TOP_K", "14"))                    # floor for the UI's top_k
    CHAT_BROAD_TOP_K: int = int(os.getenv("CHAT_BROAD_TOP_K", "20"))       # summary/gaps/contributions...
    CHAT_FETCH_MULTIPLIER: int = int(os.getenv("CHAT_FETCH_MULTIPLIER", "3"))  # over-fetch before filtering references
    CHAT_MAX_DISTANCE: float = float(os.getenv("CHAT_MAX_DISTANCE", "0.82"))   # cosine distance cutoff, 0 disables
    CHAT_HISTORY_TURNS: int = int(os.getenv("CHAT_HISTORY_TURNS", "6"))
    CHAT_CONTEXT_CHARS: int = int(os.getenv("CHAT_CONTEXT_CHARS", "24000"))    # evidence budget per prompt

    # Supabase
    SUPABASE_URL: str = os.getenv("SUPABASE_URL", "")
    SUPABASE_SERVICE_KEY: str = os.getenv("SUPABASE_SERVICE_KEY", "")

    # ChromaDB
    CHROMA_PERSIST_PATH: str = os.getenv(
        "CHROMA_PERSIST_PATH",
        str(Path(__file__).resolve().parent / "chroma_db"),
    )

    # Chunking
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "512"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "64"))

    # Web Scraping Pipeline (resilient fallback for academic source gathering, citations & case studies)
    ENABLE_SCRAPING_PIPELINE: bool = os.getenv("ENABLE_SCRAPING_PIPELINE", "True").lower() in ("true", "1", "yes")
    SCRAPER_SOURCE_GATHERING: bool = os.getenv("SCRAPER_SOURCE_GATHERING", "True").lower() in ("true", "1", "yes")
    SCRAPER_USE_CACHE: bool = os.getenv("SCRAPER_USE_CACHE", "True").lower() in ("true", "1", "yes")
    SCRAPER_MAX_RESULTS_PER_SOURCE: int = int(os.getenv("SCRAPER_MAX_RESULTS_PER_SOURCE", "5"))
    SCRAPER_ENABLE_CASE_STUDIES: bool = os.getenv("SCRAPER_ENABLE_CASE_STUDIES", "True").lower() in ("true", "1", "yes")


settings = Settings()
