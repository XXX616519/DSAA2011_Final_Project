from scipy.io import arff
import pandas as pd

data = arff.loadarff('DryBeanDataset/Dry_Bean_Dataset.arff')
df = pd.DataFrame(data[0])

df['Class'] = df['Class'].str.decode('utf-8')

missing_values = df.isnull().sum()
print("Missing value statistics:\n", missing_values)

# python Data_Preprocessing/load_data.py