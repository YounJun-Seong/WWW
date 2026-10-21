"""Shared helpers: frozen prompt rendering, seeded cached chat client (local vLLM), answer parsing, normalization."""
import hashlib
import json
import os
import re
import string
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
CACHE = os.environ.get("WQ_CACHE", f"{ROOT}/data/llm_cache")
_clients = {}


def prompt(name, **kw):
    msgs, role, buf = [], None, []
    for line in open(f"{ROOT}/prompts/{name}.txt").read().split("\n"):
        if line.startswith("### "):
            if role:
                msgs.append({"role": role, "content": "\n".join(buf).strip("\n")})
            role, buf = line[4:].strip(), []
        else:
            buf.append(line)
    msgs.append({"role": role, "content": "\n".join(buf).strip("\n")})
    for m in msgs:
        for k, v in kw.items():
            m["content"] = m["content"].replace("{" + k + "}", str(v))
    return msgs


def chat(messages, temperature, max_tokens, seed, n=1, base_url=None, model=None, extra=None):
    """Cached by full payload. base_url/model default to WQ_BASE_URL / WQ_MODEL."""
    from openai import OpenAI
    base_url = base_url or os.environ["WQ_BASE_URL"]; model = model or os.environ["WQ_MODEL"]
    payload = dict(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens, seed=seed, n=n,
                   extra=extra or {})
    key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:32]
    path = f"{CACHE}/{key}.json"
    if os.path.exists(path):
        return json.load(open(path))
    if base_url not in _clients:
        _clients[base_url] = OpenAI(base_url=base_url, api_key="EMPTY", timeout=900)
    kw = {k: v for k, v in payload.items() if k != "extra"}
    if extra:
        kw["extra_body"] = extra
    for attempt in range(6):
        try:
            r = _clients[base_url].chat.completions.create(**kw)
            break
        except Exception:
            if attempt == 5:
                raise
            time.sleep(2 * 2 ** attempt)
    rec = dict(texts=[c.message.content for c in r.choices], finish=[c.finish_reason for c in r.choices],
               input_tokens=r.usage.prompt_tokens, output_tokens=r.usage.completion_tokens)
    os.makedirs(CACHE, exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    json.dump(rec, open(tmp, "w")); os.replace(tmp, path)
    return rec


ANS = re.compile(r"^\s*\**\s*answer\s*\**\s*:\s*(.+?)\s*$", re.I | re.M)


def answer_of(text):
    """Last 'Answer: ...' line; None if absent."""
    m = ANS.findall(text or "")
    return m[-1].strip().strip("*").strip() if m else None


def normalize(s):
    """SQuAD/HotpotQA normalization: lowercase, drop punctuation, articles, extra whitespace."""
    s = (s or "").lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def f1(pred, gold):
    p, g = normalize(pred).split(), normalize(gold).split()
    common = sum(min(p.count(t), g.count(t)) for t in set(p))
    if common == 0:
        return 0.0
    pr, rc = common / len(p), common / len(g)
    return 2 * pr * rc / (pr + rc)


def fmt_passages(ps, start=1):
    return "\n".join(f"[P{start + i}] ({p['title']}) {p['text']}" for i, p in enumerate(ps))
