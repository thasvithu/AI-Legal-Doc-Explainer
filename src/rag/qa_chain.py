from __future__ import annotations
from typing import Dict, Any
from src.utils.config import AppConfig
from src.llm.gemini import GeminiClient
from src.llm.fallback import LocalLLM
from src.rag.retriever import get_retriever

from langchain_community.vectorstores import FAISS

RAG_PROMPT_PATH = "src/prompts/rag_qa.txt"

with open(RAG_PROMPT_PATH, "r", encoding="utf-8") as f:
    RAG_TEMPLATE = f.read()

class QAChain:
    def __init__(self, config: AppConfig, vs: FAISS, llm=None):
        """QAChain orchestrates retrieval + answer synthesis.

        The optional `llm` parameter allows tests / smoke scripts to inject a stub
        and skip heavyweight LocalLLM / Gemini initialization (avoids model downloads
        or large imports in constrained CI environments).
        """
        self.config = config
        self.vs = vs
        self.retriever = get_retriever(vs)
        if llm is not None:
            self.llm = llm
        else:
            if getattr(config, 'use_gemini', False):
                try:
                    self.llm = GeminiClient(config)
                except Exception:
                    self.llm = LocalLLM(config)
            else:
                self.llm = LocalLLM(config)

    def ask(self, question: str) -> Dict[str, Any]:
        import re
        # Definition query fast-path (e.g., "what is saas?", "define indemnity")
        q_norm = question.strip().lower()
        def_match = re.match(r"^(what\s+is|define|meaning\s+of)\s+([\w\-\&\/\s\.]+?)\?*$", q_norm)
        definition_target = None
        if def_match:
            # Extract candidate term, strip stopwords like 'a','the'
            raw_term = def_match.group(2).strip()
            raw_term = re.sub(r"^(the|a|an)\s+","", raw_term)
            if raw_term:
                definition_target = raw_term
        # If target looks like acronym (SaaS) normalize variant tokens
        acronyms = {"saas": ["software as a service", "saas"], "sla": ["service level agreement", "sla"], "nda": ["non-disclosure agreement", "nda"]}
        definition_hits = []
        if definition_target and hasattr(self.vs, 'docstore'):
            target_tokens = [definition_target]
            low_target = definition_target.lower()
            target_tokens.extend(acronyms.get(low_target, []))
            # Collect all sentences across docstore once
            try:
                all_docs = list(getattr(self.vs.docstore, '_dict', {}).values())
                for d in all_docs:
                    page = d.metadata.get('page')
                    for sent in re.split(r"(?<=[.!?])\s+", d.page_content):
                        s_clean = sent.strip()
                        if not (10 < len(s_clean) < 420):
                            continue
                        low = s_clean.lower()
                        if any(t in low for t in target_tokens):
                            # definitional cue words
                            if re.search(r"\b(is|means|refers to|shall mean)\b", low):
                                # score: presence of cues + proximity of term
                                score = 0
                                for t in target_tokens:
                                    if t in low:
                                        score += 4
                                if 'means' in low: score += 3
                                if 'refers to' in low: score += 2
                                if 'is' in low: score += 1
                                definition_hits.append((score, page, s_clean))
                if definition_hits:
                    definition_hits.sort(key=lambda x: (-x[0], len(x[2])))
                    top_def = definition_hits[0]
                    # Build answer; if LLM present (non-stub) refine with short prompt
                    page = top_def[1]
                    base_def = top_def[2]
                    # Heuristic trimming: remove registration / address clutter
                    trim_patterns = [
                        r",?\s*incorporated and registered in.*?\bwith company number\b.*?(?=\.|;)",
                        r",?\s*whose registered office is at.*?(?=\.|;)",
                        r"\b(company number|registration number)\s+[A-Z0-9]+",
                    ]
                    tmp = base_def
                    for pat in trim_patterns:
                        tmp = re.sub(pat, "", tmp, flags=re.I)
                    # If very long and contains 'means', shorten to clause around 'means'
                    if len(tmp) > 260 and ' means ' in tmp.lower():
                        parts = re.split(r"\bmeans\b", tmp, flags=re.I)
                        if len(parts) >= 2:
                            pre, post = parts[0], parts[1]
                            tmp = pre.strip().split(',')[0] + " means" + post.split('.')[0] + '.'
                    # Final length cap
                    if len(tmp) > 320:
                        tmp = tmp[:317].rstrip(',; ') + '...'
                    concise_def = tmp.strip()
                    citations = [{"page": page, "snippet": base_def[:300]}]
                    is_stub = isinstance(self.llm, LocalLLM) and getattr(self.llm, 'pipe', None) is None
                    if not is_stub:
                        refine_prompt = f"Provide a concise plain-language definition of '{definition_target}' grounded strictly in this contract sentence, and optionally expand acronyms. Sentence: {base_def}\nAnswer:"
                        try:
                            refined = self.llm.generate(refine_prompt).strip()
                            if 15 < len(refined) < 400:
                                return {"answer": refined, "citations": citations}
                        except Exception:
                            pass
                    # Fall back to heuristic concise version
                    if 15 < len(concise_def) < 400:
                        return {"answer": concise_def, "citations": citations}
                    return {"answer": concise_def or base_def, "citations": citations}
            except Exception:
                pass
        # 1. Retrieval
        try:
            docs = self.retriever.invoke(question)
        except AttributeError:  # older API
            docs = self.retriever.get_relevant_documents(question)

        # 2. Token prep (light stemming)
        q_low = question.lower()
        stop = {"the","a","an","is","are","to","of","and","or","in","on","for","with","does","do","shall","may","which","how","please"}
        raw_tokens = [t for t in re.findall(r"[a-zA-Z]{3,}", q_low) if t not in stop]
        def stem(t:str):
            for suf in ("ing","tion","ions","ed","es","ly","al","ment"):
                if t.endswith(suf) and len(t) > len(suf)+2:
                    return t[:-len(suf)]
            return t
        tokens = [stem(t) for t in raw_tokens]
        # synonyms map
        SYN = {"saas":["software as a service"],"terminate":["termination","end"],"payment":["fee","fees","charge"],"confidentiality":["confidential"],"liability":["liable"],"indemnity":["indemnify","indemnification"]}

        # 3. Collect candidate sentences from retrieved docs (primary) or global if few
        def collect_sentences(doc_list):
            sents = []
            for d in doc_list:
                page = d.metadata.get("page")
                for sent in re.split(r"(?<=[.!?])\s+", d.page_content):
                    s_clean = sent.strip()
                    if 12 <= len(s_clean) <= 400:
                        sents.append((page, s_clean))
            return sents
        candidates = collect_sentences(docs)
        if len(candidates) < 5 and hasattr(self.vs, 'docstore'):  # broaden
            try:
                all_docs = list(getattr(self.vs.docstore, '_dict', {}).values())
                candidates = collect_sentences(all_docs)
            except Exception:
                pass

        # 4. Scoring
        def score_sentence(s: str) -> int:
            low = s.lower()
            sc = 0
            for t in tokens:
                if t in low:
                    sc += 3
                else:
                    for syn in SYN.get(t, []):
                        if syn in low:
                            sc += 2
                            break
            # domain boosters
            for k,v in {'terminate':3,'renew':2,'payment':3,'fee':2,'confidential':2,'indemn':3,'liabil':3,'jurisdiction':2}.items():
                if k in low:
                    sc += v
            # definitional shape boost
            if any(t in low for t in tokens) and (" means " in low or " refers to " in low or low.startswith(tuple(t+" " for t in tokens))):
                sc += 4
            # length penalty (too long)
            if len(s) > 250:
                sc -= 2
            return sc

        scored = [(score_sentence(sent), page, sent) for page, sent in candidates]
        scored = [x for x in scored if x[0] > 0]

        # 5. Build context & citations
        top_sents = sorted(scored, key=lambda x: (-x[0], len(x[2])))[:5]
        if not top_sents:
            # fallback to original chunk-level prompt attempt
            context_blocks = []
            citations = []
            for d in docs:
                page = d.metadata.get("page")
                snippet = d.page_content[:280].replace("\n", " ")
                context_blocks.append(f"[Page {page}] {snippet}")
                citations.append({"page": page, "snippet": snippet})
            prompt = RAG_TEMPLATE.format(context="\n\n".join(context_blocks), question=question)
            raw_answer = self.llm.generate(prompt).strip()
            if raw_answer.lower().startswith("fallback (no local model)") or len(raw_answer) < 25:
                raw_answer = "No grounded sentence match found for the question tokens."  # final fallback
            return {"answer": raw_answer, "citations": citations}

        # Prepare answer synthesis from top sentences
        citations = []
        used_sentences = []
        seen = set()
        for sc, page, sent in top_sents:
            first = sent.split("; ")[0].strip()
            key = first.lower()
            if key in seen:
                continue
            seen.add(key)
            used_sentences.append((page, first))
            citations.append({"page": page, "snippet": sent[:300]})

        # Compose concise answer (prefer definitional then others)
        definitional = [s for p,s in used_sentences if " means " in s.lower() or " refers to " in s.lower()]
        if definitional:
            ordered = definitional + [s for p,s in used_sentences if s not in definitional]
        else:
            ordered = [s for p,s in used_sentences]
        answer = "; ".join(ordered)[:500]
        if not answer:
            answer = "No grounded sentence match found for the question tokens."
        # Confidence heuristic: average of top sentence scores normalized to 0-1 then 0-100
        if top_sents:
            raw_scores = [sc for sc,_,_ in top_sents]
            conf = min(100.0, (sum(raw_scores)/len(raw_scores))*12)  # scale factor heuristic
        else:
            conf = 0.0
        # Highlight tokens in answer
        hl_answer = answer
        for t in sorted(set(tokens), key=len, reverse=True):
            if len(t) < 3: continue
            hl_answer = re.sub(rf"\b({re.escape(t)})\b", r"**\\1**", hl_answer, flags=re.I)
        return {"answer": hl_answer, "citations": citations, "confidence": conf}


def build_qa_chain(config: AppConfig, vs: FAISS | None):
    if not vs:
        return None
    return QAChain(config, vs)
