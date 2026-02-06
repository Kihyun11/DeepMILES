import torch
import numpy as np
from model import DeepConvLSTM
from datasets import IMUDataset

def get_session_embeddings(model_path, metadata_csv, session_csv, num_classes):
    # Load the trained model
    model = DeepConvLSTM(input_channels=6, num_classes=num_classes)
    model.load_state_dict(torch.load(model_path))
    model.cuda().eval()

    # Load dataset
    ds = IMUDataset(metadata_csv, session_csv)
    
    embeddings_list = []
    with torch.no_grad():
        for i in range(len(ds)):
            data, _ = ds[i]
            data = data.unsqueeze(0).cuda()
            # Call our model with the return_embeddings flag
            emb = model(data, return_embeddings=True)
            embeddings_list.append(emb.cpu().squeeze().numpy())
            
    return np.array(embeddings_list)