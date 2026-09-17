"""
app.py
------
Daraz Customer Support Operations Assistant

- Loads a PRE-BUILT FAISS index + metadata.pkl (created by ingest.py).
  This app NEVER re-embeds or re-processes PDFs.
- Sidebar lets the agent restrict retrieval to one knowledge-base section
  (returns, delivery, refunds, sellers, payments, customer_support) or
  search across all of them.
- Uses Groq's "openai/gpt-oss-120b" model to generate answers grounded in
  the retrieved chunks.
- Reads the Groq API key from Streamlit secrets (st.secrets["GROQ_API_KEY"]) —
  never rendered in an input box.

Run with:
    streamlit run app.py

Expects a folder structure like:
    faiss_index/
        index.faiss
        metadata.pkl

Set the path via the FAISS_INDEX_DIR env var, or edit INDEX_DIR below.
"""

import os
import pickle
from pathlib import Path

import numpy as np
import streamlit as st
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
INDEX_DIR = os.environ.get("FAISS_INDEX_DIR", "faiss_index")
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"   # must match the model used in ingest.py
GROQ_MODEL = "openai/gpt-oss-120b"
TOP_K = 4

CATEGORY_LABELS = {
    "all": "All sections",
    "returns": "Returns",
    "delivery": "Delivery",
    "refunds": "Refunds",
    "sellers": "Sellers",
    "payments": "Payments",
    "customer_support": "Customer Support",
}
CATEGORY_ORDER = ["all", "returns", "delivery", "refunds", "sellers", "payments", "customer_support"]

DARAZ_ORANGE = "#F85606"
DARAZ_DARK = "#1A1A1A"


# ---------------------------------------------------------------------------
# Page config + branding
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Daraz Customer Support Operations Assistant",
    page_icon="🛍️",
    layout="wide",
)

st.markdown(
    f"""
    <style>
        .stApp {{
            background-color: #FAFAFA;
        }}
        [data-testid="stSidebar"] {{
            background-color: {DARAZ_DARK};
        }}
        [data-testid="stSidebar"] * {{
            color: #F5F5F5 !important;
        }}
        [data-testid="stSidebar"] .stRadio > label {{
            color: #F5F5F5 !important;
        }}
        .daraz-header {{
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 14px 20px;
            background: linear-gradient(90deg, {DARAZ_ORANGE} 0%, #FF8A3D 100%);
            border-radius: 10px;
            margin-bottom: 18px;
        }}
        .daraz-header h1 {{
            color: white;
            font-size: 22px;
            margin: 0;
            font-weight: 700;
        }}
        .daraz-header span {{
            color: #FFE8DA;
            font-size: 13px;
        }}
        .daraz-badge {{
            display: inline-block;
            background-color: {DARAZ_ORANGE};
            color: white;
            padding: 2px 10px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 600;
            margin-bottom: 6px;
        }}
        .source-pill {{
            display: inline-block;
            background-color: #FFF0E8;
            color: {DARAZ_ORANGE};
            border: 1px solid {DARAZ_ORANGE};
            padding: 1px 8px;
            border-radius: 10px;
            font-size: 11px;
            margin-right: 6px;
            margin-top: 4px;
        }}
        div[data-testid="stChatMessage"] {{
            border-radius: 12px;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="daraz-header">
        <h1>🛍️ Daraz Customer Support Operations Assistant</h1>
    </div>
    <span style="color:#777;font-size:13px;">
        Answers are generated only from your ingested knowledge base sections.
    </span>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Cached loaders — index/metadata/model/client are loaded once per session
# ---------------------------------------------------------------------------
def _find_index_dir(index_dir: str):
    """
    Try the configured path, then a couple of common fallback locations
    (same folder as app.py, and ./faiss_index relative to cwd), so a
    slightly-off path or a different launch directory doesn't break things.
    """
    candidates = [
        Path(index_dir),
        Path(__file__).resolve().parent / index_dir,
        Path(__file__).resolve().parent / "faiss_index",
        Path.cwd() / "faiss_index",
    ]
    for c in candidates:
        if (c / "index.faiss").exists() and (c / "metadata.pkl").exists():
            return c
    return None


@st.cache_resource(show_spinner="Loading knowledge base index...")
def load_index_and_metadata(index_dir: str):
    found_dir = _find_index_dir(index_dir)

    if found_dir is None:
        checked = "\n".join(
            f"- `{p}`" for p in [
                Path(index_dir).resolve(),
                (Path(__file__).resolve().parent / "faiss_index"),
                (Path.cwd() / "faiss_index"),
            ]
        )
        st.error(
            f"Could not find `index.faiss` / `metadata.pkl`. Checked:\n{checked}\n\n"
            "Fixes:\n"
            "1. Run `ingest.py` first to build the index, or\n"
            "2. Copy the `faiss_index` folder (containing `index.faiss` and `metadata.pkl`) "
            "into the same folder as `app.py`, or\n"
            "3. Set the `FAISS_INDEX_DIR` environment variable to the folder's absolute path "
            "(e.g. on Streamlit Cloud, `FAISS_INDEX_DIR` under Settings → Secrets, or as an env var)."
        )
        st.stop()

    index_path = found_dir / "index.faiss"
    meta_path = found_dir / "metadata.pkl"

    index = faiss.read_index(str(index_path))
    with open(meta_path, "rb") as f:
        metadata = pickle.load(f)

    return index, metadata


@st.cache_resource(show_spinner=False)
def load_embedding_model(model_name: str):
    return SentenceTransformer(model_name)


@st.cache_resource(show_spinner=False)
def get_groq_client():
    api_key = st.secrets.get("GROQ_API_KEY")
    if not api_key:
        st.error(
            "No Groq API key found. Add `GROQ_API_KEY` to your Streamlit secrets "
            "(Settings → Secrets on Streamlit Cloud, or `.streamlit/secrets.toml` locally)."
        )
        st.stop()
    return Groq(api_key=api_key)


@st.cache_resource(show_spinner=False)
def build_category_subindex(_index, category: str, _metadata):
    """
    Build a small FAISS index containing only the vectors belonging to one
    category, reconstructed from the main (already-built) index. Cached per
    category so this only happens once per session, not per query.
    Returns (subindex, id_map) where id_map[i] = original metadata index.
    """
    ids = [i for i, m in enumerate(_metadata) if m["category"] == category]
    if not ids:
        return None, []

    dim = _index.d
    vectors = np.zeros((len(ids), dim), dtype="float32")
    for row, orig_id in enumerate(ids):
        vectors[row] = _index.reconstruct(orig_id)

    subindex = faiss.IndexFlatIP(dim)
    subindex.add(vectors)
    return subindex, ids


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
def retrieve_chunks(query: str, category: str, index, metadata, embed_model, k: int = TOP_K):
    query_vec = embed_model.encode([query], normalize_embeddings=True).astype("float32")

    if category == "all":
        scores, ids = index.search(query_vec, k)
        ids = ids[0]
        scores = scores[0]
        results = [
            {**metadata[i], "score": float(s)}
            for i, s in zip(ids, scores) if i != -1
        ]
    else:
        subindex, id_map = build_category_subindex(index, category, metadata)
        if subindex is None:
            return []
        scores, sub_ids = subindex.search(query_vec, min(k, subindex.ntotal))
        results = [
            {**metadata[id_map[i]], "score": float(s)}
            for i, s in zip(sub_ids[0], scores[0]) if i != -1
        ]

    return results


# ---------------------------------------------------------------------------
# LLM generation
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are the Daraz Customer Support Operations Assistant.
You help support agents quickly find accurate answers from Daraz's internal
policy knowledge base (returns, delivery, refunds, sellers, payments, customer support).

Rules:
- Answer ONLY using the provided context chunks. Do not invent policy details.
- If the context does not contain the answer, say clearly that the knowledge
  base does not cover this and suggest escalating to the relevant team.
- Be concise, structured, and use bullet points for steps or conditions when helpful.
- When relevant, mention which knowledge-base section(s) the answer is based on.
"""


def generate_answer(client, query: str, chunks: list):
    if not chunks:
        context_block = "No relevant context was found in the knowledge base."
    else:
        context_block = "\n\n".join(
            f"[Section: {c['category']}] (source: {Path(c['source']).name})\n{c['text']}"
            for c in chunks
        )

    user_prompt = f"""Context from the knowledge base:
{context_block}

Agent question: {query}

Answer the agent's question using only the context above."""

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Sidebar — section selector
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<span class="daraz-badge">KNOWLEDGE BASE</span>', unsafe_allow_html=True)
    st.markdown("### Search Scope")
    st.caption("Restrict answers to a single policy section, or search everything.")

    selected_label = st.radio(
        "Section",
        options=[CATEGORY_LABELS[c] for c in CATEGORY_ORDER],
        index=0,
        label_visibility="collapsed",
    )
    selected_category = CATEGORY_ORDER[
        [CATEGORY_LABELS[c] for c in CATEGORY_ORDER].index(selected_label)
    ]

    st.divider()
    top_k = st.slider("Chunks to retrieve", min_value=2, max_value=8, value=TOP_K)

    st.divider()
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.caption("Powered by Groq · openai/gpt-oss-120b")


# ---------------------------------------------------------------------------
# Load resources
# ---------------------------------------------------------------------------
index, metadata = load_index_and_metadata(INDEX_DIR)
embed_model = load_embedding_model(EMBEDDING_MODEL_NAME)
groq_client = get_groq_client()

with st.sidebar:
    st.caption(f"📚 {len(metadata)} chunks loaded")


# ---------------------------------------------------------------------------
# Chat state + history
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            pills = "".join(
                f'<span class="source-pill">{CATEGORY_LABELS.get(s["category"], s["category"])} · '
                f'{Path(s["source"]).name}</span>'
                for s in msg["sources"]
            )
            st.markdown(pills, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------
placeholder = (
    "Ask about Daraz policies (all sections)..."
    if selected_category == "all"
    else f"Ask about {CATEGORY_LABELS[selected_category]} policies..."
)

if prompt := st.chat_input(placeholder):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base..."):
            chunks = retrieve_chunks(prompt, selected_category, index, metadata, embed_model, k=top_k)
        with st.spinner("Generating answer..."):
            answer = generate_answer(groq_client, prompt, chunks)

        st.markdown(answer)
        if chunks:
            pills = "".join(
                f'<span class="source-pill">{CATEGORY_LABELS.get(c["category"], c["category"])} · '
                f'{Path(c["source"]).name}</span>'
                for c in chunks
            )
            st.markdown(pills, unsafe_allow_html=True)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": chunks,
    })
