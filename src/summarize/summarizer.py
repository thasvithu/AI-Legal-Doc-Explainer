from __future__ import annotations
from typing import List, Dict
from src.utils.config import AppConfig
from src.llm.gemini import GeminiClient
from src.llm.fallback import LocalLLM
from src.utils.types import Document, Chunk

SUM_PROMPT_PATH = "src/prompts/summarization.txt"
with open(SUM_PROMPT_PATH, "r", encoding="utf-8") as f:
    SUM_TEMPLATE = f.read()


def _get_llm(config: AppConfig):
    if config.use_gemini:
        try:
            return GeminiClient(config)
        except Exception:
            return LocalLLM(config)
    return LocalLLM(config)


def heuristic_document_summary(text_blocks: List[str]) -> str:
    """Public reusable heuristic summary (fast, no LLM)."""
    import re
    if not text_blocks:
        return "- (No text extracted)"
    joined = " \n".join(text_blocks)
    sentences = re.split(r'(?<=[.!?])\s+', joined)
    categories = {
        'Parties/Purpose': ['party', 'parties', 'purpose', 'provide', 'service', 'agreement'],
        'Term & Renewal': ['term', 'renew', 'expiration', 'renewal', 'duration'],
        'Payment & Fees': ['payment', 'fee', 'invoice', 'pricing', 'charges', 'payable'],
        'Data & Privacy': ['data', 'personal', 'privacy', 'gdpr', 'processing', 'controller', 'processor'],
        'Confidentiality & IP': ['confidential', 'secret', 'ip ', 'intellectual', 'license', 'licence', 'ownership'],
        'Liability & Indemnity': ['liability', 'indemn', 'limit', 'cap', 'damages'],
        'Termination': ['terminate', 'termination', 'notice', 'breach', 'suspend'],
        'Warranties & Disclaimers': ['warrant', 'disclaim', 'as is'],
        'Dispute / Law': ['jurisdiction', 'govern', 'law', 'dispute', 'arbitr', 'court'],
        'Risks / Unusual': ['auto-renew', 'penalt', 'liquidated', 'sole discretion', 'unilateral']
    }
    all_keywords = {kw for kws in categories.values() for kw in kws}
    scored = []
    for s in sentences:
        low = s.lower()
        score = sum(1 for k in all_keywords if k in low)
        if 15 < len(s) < 300 and score:
            scored.append((score, s.strip()))
    scored.sort(key=lambda x: (-x[0], len(x[1])))
    category_best = {}
    for _, sent in scored:
        low = sent.lower()
        for cat, kws in categories.items():
            if any(k in low for k in kws):
                if cat not in category_best:
                    category_best[cat] = sent
                break
        if len(category_best) >= 10:
            break
    order = [
        'Parties/Purpose', 'Term & Renewal', 'Payment & Fees', 'Termination',
        'Data & Privacy', 'Confidentiality & IP', 'Liability & Indemnity',
        'Warranties & Disclaimers', 'Dispute / Law', 'Risks / Unusual'
    ]
    bullets = []
    for cat in order:
        if cat in category_best:
            txt = category_best[cat]
            txt = re.sub(r'\b(the|a|an)\b\s+', '', txt, flags=re.I)
            if len(txt) > 170:
                txt = txt[:167] + '...'
            bullets.append(f"- {cat}: {txt.rstrip('. ')}.")
    if not bullets:
        for _, s in scored[:8]:
            bullets.append('- ' + s)
    return "\n".join(bullets[:10])


def summarize_documents(config: AppConfig, docs: List[Document], chunks: List[Chunk]) -> Dict[str, Dict[str, str]]:
    llm = _get_llm(config)
    summaries: Dict[str, Dict[str, str]] = {}
    chunks_by_doc: Dict[str, List[str]] = {}
    for c in chunks:
        chunks_by_doc.setdefault(c.document_name, []).append(c.content)

    for doc in docs:
        parts = chunks_by_doc.get(doc.name, [])
        use_heuristic = isinstance(llm, LocalLLM) and getattr(llm, 'pipe', None) is None
        if use_heuristic:
            summaries[doc.name] = {"bullets": heuristic_document_summary(parts)}
            continue

        bullet_accum: List[str] = []
        for i in range(0, len(parts), 6):
            batch = parts[i:i+6]
            prompt = SUM_TEMPLATE.format(text="\n\n".join(batch))
            resp = llm.generate(prompt)
            bullet_accum.append(resp.strip())
        overall_prompt = (
            "You will be given bullet lists extracted from a legal agreement. Consolidate them into 5-10 NEW, UNIQUE, plain-language bullets (each starting with '- '). Focus on: parties & purpose, key obligations, payment & fees, term & renewal/termination, liability & indemnity, confidentiality/IP, jurisdiction/dispute, unusual penalties or auto-renewal traps. Avoid repetition; no legalese; <=25 words per bullet.\n\n" + "\n".join(bullet_accum)
        )
        overall = llm.generate(overall_prompt)
        # Normalize bullet formatting
        lines = [l.strip('- ').strip() for l in overall.splitlines() if l.strip()]
        # If model ignored structure, attempt category mapping
        categories_map = {
            'parties': 'Parties/Purpose', 'purpose': 'Parties/Purpose', 'term': 'Term & Renewal', 'renew': 'Term & Renewal',
            'payment': 'Payment & Fees', 'fee': 'Payment & Fees', 'invoice': 'Payment & Fees', 'data': 'Data & Privacy',
            'privacy': 'Data & Privacy', 'confidential': 'Confidentiality & IP', 'ip ': 'Confidentiality & IP', 'intellectual': 'Confidentiality & IP',
            'indemn': 'Liability & Indemnity', 'liability': 'Liability & Indemnity', 'terminate': 'Termination', 'notice': 'Termination',
            'warrant': 'Warranties & Disclaimers', 'disclaim': 'Warranties & Disclaimers', 'jurisdiction': 'Dispute / Law', 'law': 'Dispute / Law',
            'arbitr': 'Dispute / Law', 'auto-renew': 'Risks / Unusual', 'penalt': 'Risks / Unusual', 'sole discretion': 'Risks / Unusual'
        }
        cat_best = {}
        import re
        for l in lines:
            low = l.lower()
            for k, cat in categories_map.items():
                if k in low:
                    if cat not in cat_best:
                        cat_best[cat] = l
                    break
        if 0 < len(cat_best) <= 10:
            order = ['Parties/Purpose','Term & Renewal','Payment & Fees','Termination','Data & Privacy','Confidentiality & IP','Liability & Indemnity','Warranties & Disclaimers','Dispute / Law','Risks / Unusual']
            lines = [f"{cat}: {cat_best[cat]}" for cat in order if cat in cat_best]
        cleaned = []
        seen = set()
        for l in lines:
            if not l:
                continue
            key = l.lower()
            if key in seen:
                continue
            seen.add(key)
            cleaned.append('- ' + l[:160])
            if len(cleaned) >= 10:
                break
        summaries[doc.name] = {"bullets": "\n".join(cleaned)}
    return summaries
