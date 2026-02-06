import pandas as pd
from sklearn.model_selection import train_test_split

def make_split(actions_csv_path, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    # 1. Load the trimming and labeling info
    df = pd.read_csv(actions_csv_path)
    
    # 2. Identify unique sessions to ensure a clean split
    unique_sessions = df['session_id'].unique()
    
    # 3. Perform the split (80% Train, 10% Val, 10% Test)
    train_ids, temp_ids = train_test_split(
        unique_sessions, 
        train_size=train_ratio, 
        random_state=42,
        shuffle=True
    )
    
    # Split the remaining 20% into two equal halves (10% + 10%)
    val_ids, test_ids = train_test_split(
        temp_ids, 
        train_size=0.5, 
        random_state=42
    )
    
    # 4. Create Sub-Dataframes
    train_split = df[df['session_id'].isin(train_ids)]
    val_split = df[df['session_id'].isin(val_ids)]
    test_split = df[df['session_id'].isin(test_ids)]
    
    # 5. Save the split files
    train_split.to_csv('metadata_train.csv', index=False)
    val_split.to_csv('metadata_val.csv', index=False)
    test_split.to_csv('metadata_test.csv', index=False)
    
    print(f"Split complete!")
    print(f"Train: {len(train_split)} segments | Val: {len(val_split)} segments | Test: {len(test_split)} segments")

# Usage:
make_split("C:/Users/User/Documents/GitHub/Deep-MILES-Personalized-Performance-Evaluation-AI-Model-for-Next-Gen-KCTC/action_labels.csv")