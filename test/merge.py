import pandas as pd
train_dataset = pd.read_csv(r"D:\File\Preprocessed_data\train_set.csv")
val_dataset = pd.read_csv(r"D:\File\Preprocessed_data\val_set.csv")
test_dataset = pd.read_csv(r"D:\File\Preprocessed_data\test_set.csv")
journal_dataset = pd.read_csv(r"D:\File\Preprocessed_data\journal_category.csv")

print("Load datasets successfully!")

df = pd.merge(train_dataset, journal_dataset, on='Label', how='left')
df.to_csv(r"D:\File\Preprocessed_data\train_set_merged.csv", index=False)

df2 = pd.merge(val_dataset, journal_dataset, on='Label', how='left')
df2.to_csv(r"D:\File\Preprocessed_data\val_set_merged.csv", index=False)

df3 = pd.merge(test_dataset, journal_dataset, on='Label', how='left')
df3.to_csv(r"D:\File\Preprocessed_data\test_set_merged.csv", index=False)

print("Merge datasets successfully!")