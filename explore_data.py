import pandas as pd

df = pd.read_csv('train.csv')

print("--- First 5 rows of the dataset ---")
print(df.head())

# 3. See how many examples of each type of abuse we have
print("\n--- Count of each label ---")
print("Toxic:", df['toxic'].sum())
print("Severe Toxic:", df['severe_toxic'].sum())
print("Obscene:", df['obscene'].sum())
print("Threat:", df['threat'].sum())
print("Insult:", df['insult'].sum())
print("Identity Hate:", df['identity_hate'].sum())