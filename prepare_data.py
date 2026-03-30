import pandas as pd

print("Loading dataset...")
df = pd.read_csv('train.csv')

abuse_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
df['is_abusive'] = df[abuse_columns].sum(axis=1) > 0

abusive_comments = df[df['is_abusive'] == True]
neutral_comments = df[df['is_abusive'] == False]

# THE FIX: Keep 3 times as many neutral comments as abusive ones
target_neutral_count = len(abusive_comments) * 3
neutral_downsampled = neutral_comments.sample(n=target_neutral_count, random_state=42)

balanced_df = pd.concat([abusive_comments, neutral_downsampled])
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
balanced_df = balanced_df.drop(columns=['is_abusive'])

print(f"\nNew 1:3 Balanced Dataset Total Rows: {len(balanced_df)}")
balanced_df.to_csv('better_train.csv', index=False)
print("Success: Saved to 'better_train.csv'")