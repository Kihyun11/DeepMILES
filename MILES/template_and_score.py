# template_and_score.py
# Goal C: Build canonical (modal) action templates from TRAIN, then score labeled VAL/TEST/UNSEEN sessions.
# Also prints session-level classification results (true vs predicted label).

import numpy as np
import pandas as pd
from collections import defaultdict

from scipy.spatial.distance import cosine
from fastdtw import fastdtw

from emb_ext import extract_embeddings_by_session


# -----------------------------
# Utilities
# -----------------------------
def _read_metadata(metadata_csv: str) -> pd.DataFrame:
    md = pd.read_csv(metadata_csv)
    md.columns = md.columns.str.strip()
    md["session_id"] = md["session_id"].astype(str)  # normalize to str to match embedding dict keys
    md["label"] = md["label"].astype(str)
    return md


def _mean_embedding(seq: np.ndarray) -> np.ndarray:
    # seq: (num_windows, emb_dim)
    return np.mean(seq, axis=0)


def _embedding_smoothness(seq: np.ndarray) -> float:
    """
    Measures how smoothly the embedding changes across time.
    Returns score in [0,1], higher is smoother.
    """
    if seq is None or len(seq) < 2:
        return 1.0

    sims = []
    for i in range(1, len(seq)):
        # cosine similarity between consecutive embeddings
        sims.append(1.0 - cosine(seq[i - 1], seq[i]))
    sims = np.array(sims, dtype=np.float64)

    return float(np.clip(np.mean(sims), 0.0, 1.0))


def _temporal_consistency_dtw(seq: np.ndarray, template_seq: np.ndarray) -> float:
    """
    DTW over embedding sequences with cosine distance.
    Returns score in (0,1], higher is better.
    """
    if seq is None or template_seq is None or len(seq) == 0 or len(template_seq) == 0:
        return 0.0

    dist, path = fastdtw(seq, template_seq, dist=cosine)
    norm = dist / max(len(path), 1)
    return float(1.0 / (1.0 + norm))


# -----------------------------
# Label mapping (important for predicted labels)
# -----------------------------
def build_label_id_to_name(metadata_train_csv: str):
    """
    Builds id->label mapping consistent with your dataset.py label_map logic:
    enumerate(metadata_train['label'].unique()) in first-appearance order.
    """
    md = pd.read_csv(metadata_train_csv)
    md.columns = md.columns.str.strip()
    labels = list(pd.Series(md["label"].astype(str)).unique())
    return {i: lab for i, lab in enumerate(labels)}


def session_prediction_from_probs(prob_seq: np.ndarray):
    """
    prob_seq: (num_windows, num_classes)
    Aggregates window probs to a session prediction.
    Returns: pred_id, pred_conf
    """
    if prob_seq is None or len(prob_seq) == 0:
        return None, None
    avg = np.mean(prob_seq, axis=0)
    pred_id = int(np.argmax(avg))
    pred_conf = float(np.max(avg))
    return pred_id, pred_conf


# -----------------------------
# Build templates from TRAIN
# -----------------------------
def build_canonical_templates(
    train_emb_by_session: dict,
    metadata_train_csv: str,
    k_exemplars: int = 1,
):
    """
    Returns:
      templates[label] = {
         "template_vec": (emb_dim,),
         "template_seq": (num_windows, emb_dim),
         "exemplar_session_ids": [...],
         "num_train_sessions": int
      }
    """
    md = _read_metadata(metadata_train_csv)

    # Group session ids by label (only those that exist in embeddings)
    label_to_sids = defaultdict(list)
    for _, row in md.iterrows():
        sid = row["session_id"]
        lab = row["label"]
        if sid in train_emb_by_session:
            label_to_sids[lab].append(sid)

    templates = {}

    for lab, sids in label_to_sids.items():
        if len(sids) == 0:
            continue

        # Per-session mean embeddings
        session_means = []
        for sid in sids:
            seq = train_emb_by_session[sid]
            session_means.append(_mean_embedding(seq))
        session_means = np.stack(session_means, axis=0)  # (num_sessions, emb_dim)

        # Canonical (modal) embedding vector = average of session means
        template_vec = np.mean(session_means, axis=0)

        # Pick exemplars: sessions whose mean embedding is closest to template_vec
        dists = []
        for sid in sids:
            mu = _mean_embedding(train_emb_by_session[sid])
            d = cosine(mu, template_vec)  # smaller = closer
            dists.append((d, sid))
        dists.sort(key=lambda x: x[0])

        exemplar_sids = [sid for _, sid in dists[:max(1, k_exemplars)]]

        # For DTW template, use the best exemplar sequence
        template_seq = train_emb_by_session[exemplar_sids[0]]

        templates[lab] = {
            "template_vec": template_vec,
            "template_seq": template_seq,
            "exemplar_session_ids": exemplar_sids,
            "num_train_sessions": len(sids),
        }

    return templates


# -----------------------------
# Score labeled VAL/TEST/Unseen sessions
# -----------------------------
def score_labeled_sessions_against_templates(
    emb_by_session: dict,
    prob_by_session: dict,
    metadata_csv: str,
    templates: dict,
    id_to_label: dict,
) -> pd.DataFrame:
    md = _read_metadata(metadata_csv)

    rows = []
    for _, row in md.iterrows():
        sid = row["session_id"]
        true_lab = row["label"]

        # Must have embeddings for this session
        if sid not in emb_by_session:
            continue

        # Must have template for this true label
        if true_lab not in templates:
            continue

        seq = emb_by_session[sid]
        mu = _mean_embedding(seq)

        # --- session-level prediction from probs ---
        prob_seq = prob_by_session.get(sid, None) if prob_by_session is not None else None
        pred_id, pred_conf = session_prediction_from_probs(prob_seq)
        pred_lab = id_to_label.get(pred_id, f"class_{pred_id}") if pred_id is not None else None
        is_correct = (pred_lab == true_lab) if pred_lab is not None else None

        # --- template comparison (Goal C) ---
        template_vec = templates[true_lab]["template_vec"]
        template_seq = templates[true_lab]["template_seq"]

        # 1) Form similarity (session mean vs canonical vector)
        form_sim = 1.0 - cosine(mu, template_vec)

        # 2) Temporal consistency (DTW on embedding sequences)
        temporal_cons = _temporal_consistency_dtw(seq, template_seq)

        # 3) Stability inside the session (embedding smoothness)
        smooth = _embedding_smoothness(seq)

        rows.append({
            "session_id": sid,
            "true_label": true_lab,
            "pred_label": pred_lab,
            "pred_conf": pred_conf,
            "is_correct": is_correct,
            "num_windows": int(seq.shape[0]),
            "form_similarity": float(form_sim),
            "temporal_consistency": float(temporal_cons),
            "embedding_smoothness": float(smooth),
            "train_sessions_for_label": templates[true_lab]["num_train_sessions"],
            "template_exemplar_session": templates[true_lab]["exemplar_session_ids"][0],
        })

    df = pd.DataFrame(rows)
    if len(df) > 0:
        # Weighted overall score for the evaluated session (NOT template)
        df["overall_score"] = (
            0.5 * df["form_similarity"] +
            0.3 * df["temporal_consistency"] +
            0.2 * df["embedding_smoothness"]
        )
    return df


# -----------------------------
# Main runner
# -----------------------------
def run_goal_c(
    metadata_train: str = "metadata_train.csv",
    metadata_eval: str = "metadata_val.csv",  # set to metadata_test.csv when needed
    session_csv: str = "session.csv",
    model_path: str = None,                   # None => cfg.MODEL_SAVE_PATH in emb_ext.py
    out_csv: str = "scores_eval.csv",
):
    # 1) Extract TRAIN embeddings + probs (probs not required for templates but returned anyway)
    train_emb, train_prob = extract_embeddings_by_session(
        model_path=model_path,
        metadata_csv=metadata_train,
        session_csv=session_csv,
        save_npz=None,
        save_probs_npz=None,
    )

    # 2) Build canonical templates per label (from TRAIN)
    templates = build_canonical_templates(train_emb, metadata_train, k_exemplars=1)

    # 3) Build id->label mapping consistent with training label order
    id_to_label = build_label_id_to_name(metadata_train)

    # 4) Extract EVAL embeddings + probs (VAL/TEST/UNSEEN labeled)
    eval_emb, eval_prob = extract_embeddings_by_session(
        model_path=model_path,
        metadata_csv=metadata_eval,
        session_csv=session_csv,
        save_npz=None,
        save_probs_npz=None,
    )

    # 5) Score sessions
    df = score_labeled_sessions_against_templates(
        emb_by_session=eval_emb,
        prob_by_session=eval_prob,
        metadata_csv=metadata_eval,
        templates=templates,
        id_to_label=id_to_label,
    )

    if df is None or len(df) == 0:
        print("[WARN] No sessions were scored. Check: session_id matching, file paths, and label coverage.")
        return

    # 6) Save
    df.to_csv(out_csv, index=False)
    print(f"\n[OK] Saved scores: {out_csv}\n")

    # 7) Print classification results (scores for evaluated data)
    print("===== CLASSIFICATION (SESSION-LEVEL) =====")
    print(
        df[["session_id", "true_label", "pred_label", "pred_conf", "is_correct"]]
        .sort_values(["true_label", "pred_conf"], ascending=[True, False])
        .to_string(index=False, float_format=lambda x: f"{x:.4f}")
    )

    acc = df["is_correct"].dropna().mean()
    print(f"\nSession-level accuracy: {acc*100:.2f}%\n")

    # 8) Print quality / stability scores for evaluated sessions
    print("===== EVALUATED SESSION QUALITY SCORES =====")
    cols = [
        "session_id", "true_label", "num_windows",
        "form_similarity", "temporal_consistency", "embedding_smoothness", "overall_score"
    ]
    df_print = df.sort_values(["true_label", "overall_score"], ascending=[True, False])
    print(df_print[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    # Change metadata_eval to "metadata_test.csv" for test scoring
    run_goal_c(
        metadata_train="metadata_train.csv",
        metadata_eval="metadata_val.csv",
        session_csv="session.csv",
        out_csv="scores_val.csv",
    )
