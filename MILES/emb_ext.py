import torch
import numpy as np
from collections import defaultdict

import configs as cfg
from model import DeepConvLSTM
from dataset import IMUDataset


def extract_embeddings_by_session(
    model_path: str = None,
    metadata_csv: str = None,
    session_csv: str = None,
    num_classes: int = None,
    save_npz: str = None,
    save_probs_npz: str = None,
):
    model_path = model_path or cfg.MODEL_SAVE_PATH
    metadata_csv = metadata_csv or cfg.VAL_METADATA
    session_csv = session_csv or cfg.SESSION_CSV
    num_classes = num_classes or cfg.NUM_CLASSES

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DeepConvLSTM(
        input_channels=cfg.INPUT_CHANNELS,
        num_classes=num_classes,
        conv_kernels=cfg.CONV_KERNELS,
        lstm_units=cfg.LSTM_UNITS,
    )

    # You can set weights_only=True if your torch version supports it:
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()

    ds = IMUDataset(metadata_csv, session_csv, cfg.WINDOW_SIZE, cfg.STEP_SIZE)

    by_session_emb = defaultdict(list)
    by_session_prob = defaultdict(list)

    with torch.no_grad():
        for i in range(len(ds)):
            x, _, sid = ds[i]
            x = x.unsqueeze(0).to(device)  # (1, 1, T, 6)

            # embeddings
            emb = model(x, return_embeddings=True)  # (1, 128)
            by_session_emb[sid].append(emb.squeeze(0).cpu().numpy())

            # classification probabilities
            logits = model(x)  # (1, C)
            probs = torch.softmax(logits, dim=1)  # (1, C)
            by_session_prob[sid].append(probs.squeeze(0).cpu().numpy())

    # Convert to dict of arrays
    by_session_emb = {str(k): np.stack(v, axis=0) for k, v in by_session_emb.items()}
    by_session_prob = {str(k): np.stack(v, axis=0) for k, v in by_session_prob.items()}

    if save_npz:
        np.savez(save_npz, **by_session_emb)
        print(f"[OK] Saved embeddings: {save_npz} (keys=session_id)")

    if save_probs_npz:
        np.savez(save_probs_npz, **by_session_prob)
        print(f"[OK] Saved probs: {save_probs_npz} (keys=session_id)")

    return by_session_emb, by_session_prob


if __name__ == "__main__":
    extract_embeddings_by_session()
