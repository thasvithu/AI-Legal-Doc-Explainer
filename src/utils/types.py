from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class Document:
    name: str
    text: str
    pages: int
    pages_text: List[str] = field(default_factory=list)  # raw text per page (cleaned) for accurate citation mapping

@dataclass
class Chunk:
    id: str
    document_name: str
    page: int
    content: str

@dataclass
class ClauseResult:
    clause_type: str
    explanation: str
    snippet: str
    page: int
    importance: str

@dataclass
class RedFlagResult:
    risk_type: str
    reason: str
    snippet: str
    confidence: float
    page: int

QAHistory = List[Dict[str, Any]]
