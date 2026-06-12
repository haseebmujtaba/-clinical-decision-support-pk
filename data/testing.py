import pandas as pd
df = pd.read_csv("data/merged_training_data.csv")

# Pima Diabetes dataset = all female. Whichever sex code dominates
# the diabetes rows tells you which number = F.
pima_rows = df[df["diagnosis"].isin(["Diabetes", "No Diabetes", "Type 2 Diabetes"])]
print(pima_rows["sex"].value_counts())