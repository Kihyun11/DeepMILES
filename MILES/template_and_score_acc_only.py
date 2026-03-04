# template_and_score_acc_only.py
from template_and_score import (
    build_label_id_to_name,
    build_canonical_templates,
    score_labeled_sessions_against_templates,
)
from emb_ext_acc_only import extract_embeddings_by_session_acc_only

def run_goal_c_acc_only(
    metadata_train: str = "metadata_train.csv",
    metadata_eval: str = "metadata_test.csv",
    session_csv: str = "session.csv",
    model_path: str = "model_acc_only.pth",
    out_csv: str = "scores_acc_only.csv",
):
    # 1) Extract TRAIN embeddings + probs
    train_emb, train_prob = extract_embeddings_by_session_acc_only(
        model_path=model_path,
        metadata_csv=metadata_train,
        session_csv=session_csv,
        save_npz=None,
        save_probs_npz=None,
    )

    # 2) Templates
    templates = build_canonical_templates(train_emb, metadata_train, k_exemplars=1)

    # 3) id->label mapping
    id_to_label = build_label_id_to_name(metadata_train)

    # 4) Extract EVAL embeddings + probs
    eval_emb, eval_prob = extract_embeddings_by_session_acc_only(
        model_path=model_path,
        metadata_csv=metadata_eval,
        session_csv=session_csv,
        save_npz=None,
        save_probs_npz=None,
    )

    # 5) Score + classification print/save (same function)
    df = score_labeled_sessions_against_templates(
        emb_by_session=eval_emb,
        prob_by_session=eval_prob,
        metadata_csv=metadata_eval,
        templates=templates,
        id_to_label=id_to_label,
    )

    if df is None or len(df) == 0:
        print("[ACC-ONLY][WARN] No sessions were scored. Check session_id matching & paths.")
        return

    df.to_csv(out_csv, index=False)
    print(f"\n[ACC-ONLY][OK] Saved scores: {out_csv}\n")

    print("===== ACC-ONLY CLASSIFICATION (SESSION-LEVEL) =====")
    print(df[["session_id","true_label","pred_label","pred_conf","is_correct"]]
          .to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    acc = df["is_correct"].dropna().mean()
    print(f"\n[ACC-ONLY] Session-level accuracy: {acc*100:.2f}%\n")

    print("===== ACC-ONLY QUALITY SCORES =====")
    cols = ["session_id","true_label","num_windows",
            "form_similarity","temporal_consistency","embedding_smoothness","overall_score"]
    print(df[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

if __name__ == "__main__":
    run_goal_c_acc_only(
        metadata_train="metadata_train.csv",
        metadata_eval="metadata_test.csv",
        session_csv="session.csv",
        model_path="model_acc_only.pth",
        out_csv="scores_acc_only.csv",
    )