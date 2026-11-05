"""
embeddings/embed_sentence_transformer.py  —  Representation A
=============================================================
Serialise each row to a "col=value; col=value; ..." string and encode with a
small SentenceTransformer (all-MiniLM-L6-v2, 384-d). No fitting — encoding is
row-independent. Expected the weakest: numbers get mangled by text tokenisation.
Requires internet on first run to download the model.
"""
import numpy as np


def build(df, cat, cont, train_mask, y, cfg) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    cols = cat + cont
    sub = df[cols].astype(str)
    texts = [
        "; ".join(f"{c}={v}" for c, v in zip(cols, row))
        for row in sub.itertuples(index=False, name=None)
    ]

    model_name = cfg["representations"]["sentence_transformer"]["model_name"]
    st = SentenceTransformer(model_name)
    Z = st.encode(texts, batch_size=256, convert_to_numpy=True, show_progress_bar=True)
    print(f"    [sentence-transformer] output dim = {Z.shape[1]}")
    return Z.astype(np.float32)
