# emb_ext_acc_only.py
import torch
import numpy as np
from collections import defaultdict

import configs as cfg
from model import DeepConvLSTM
from dataset_acc_only import IMUDatasetAccOnly

def extract_embeddings_by_session_acc_only(
    model_path: str = "model_acc_only.pth",
    metadata_csv: str = None,
    session_csv: str = None,
    num_classes: int = None,
    save_npz: str = None,
    save_probs_npz: str = None,
):
    metadata_csv = metadata_csv or cfg.VAL_METADATA
    session_csv = session_csv or cfg.SESSION_CSV
    num_classes = num_classes or cfg.NUM_CLASSES

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DeepConvLSTM(
        input_channels=3,
        num_classes=num_classes,
        conv_kernels=cfg.CONV_KERNELS,
        lstm_units=cfg.LSTM_UNITS,
    )

    # safer torch.load (optional)
    try:
        state = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(model_path, map_location=device)

    model.load_state_dict(state)
    model.to(device).eval()

    ds = IMUDatasetAccOnly(metadata_csv, session_csv, cfg.WINDOW_SIZE, cfg.STEP_SIZE)

    emb_by_session = defaultdict(list)
    prob_by_session = defaultdict(list)

    with torch.no_grad():
        for i in range(len(ds)):
            x, _y, sid = ds[i]
            x = x.unsqueeze(0).to(device)  # (1, 1, T, 3)

            emb = model(x, return_embeddings=True)  # (1, lstm_units)
            logits = model(x, return_embeddings=False)  # (1, num_classes)
            probs = torch.softmax(logits, dim=1)

            emb_by_session[str(sid)].append(emb.squeeze(0).cpu().numpy())
            prob_by_session[str(sid)].append(probs.squeeze(0).cpu().numpy())

    emb_by_session = {k: np.stack(v, axis=0) for k, v in emb_by_session.items()}
    prob_by_session = {k: np.stack(v, axis=0) for k, v in prob_by_session.items()}

    if save_npz:
        np.savez(save_npz, **emb_by_session)
        print(f"[ACC-ONLY] Saved embeddings: {save_npz}")
    if save_probs_npz:
        np.savez(save_probs_npz, **prob_by_session)
        print(f"[ACC-ONLY] Saved probs: {save_probs_npz}")

    return emb_by_session, prob_by_session

if __name__ == "__main__":
    extract_embeddings_by_session_acc_only()