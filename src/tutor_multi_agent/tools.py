from __future__ import annotations

from datetime import datetime

import requests
from langchain_core.tools import tool


@tool
def get_poem_tool() -> str:
    """Fetch one random Chinese poem sentence from Jinrishici API."""
    url = "https://v1.jinrishici.com/all.json"
    try:
        response = requests.get(url, timeout=8)
        response.raise_for_status()
        data = response.json()
        title = data.get("origin", "Unknown title")
        dynasty = data.get("category", "Unknown dynasty")
        author = data.get("author", "Unknown author")
        content = data.get("content", "").strip()
        return f"{content}\n— {author}, {title} ({dynasty})"
    except Exception as exc:  # noqa: BLE001
        return f"Failed to fetch poem. Error: {exc}"


@tool
def get_date_tool() -> str:
    """Return current local datetime in ISO format."""
    return datetime.now().isoformat(timespec="seconds")
