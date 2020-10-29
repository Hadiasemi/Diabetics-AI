import pandas as pd




sheet_id="1YL4jVF19FypUYeBjcQERw6NlmGV6JPV-Ketp1Ql56N8"
df=pd.read_csv(f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv")
print(df.head())