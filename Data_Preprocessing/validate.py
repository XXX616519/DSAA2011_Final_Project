import matplotlib.pyplot as plt
from scipy.io import arff
import pandas as pd

data = arff.loadarff('DryBeanDataset/Dry_Bean_Dataset.arff')
df = pd.DataFrame(data[0])

df['Class'] = df['Class'].str.decode('utf-8')

class_dist = df['Class'].value_counts()

print("\nCategory Distribution Statistics:")
print(class_dist)


plt.figure(figsize=(10,6))
class_dist.plot(kind='bar')
plt.title("Class Distribution Before Processing")
plt.xticks(rotation=45)
plt.savefig('DryBeanDataset/class_distribution.png')
plt.show()

# python Data_Preprocessing/validate.py