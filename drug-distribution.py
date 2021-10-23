import pandas as pd
import matplotlib.pyplot as plt
from sklearn import model_selection as ms

data = pd.read_csv('drug200.csv')

print(data)
counts = data["Drug"].value_counts()
print(counts.index)
f1 = plt.figure(1)
plt.hist(data["Drug"])
f2 = plt.figure(2)
plt.bar(range(1, 6), counts, tick_label=list(counts.index))
plt.xlabel("Drugs")
plt.ylabel("Frequency")
plt.title("Distribution")
# plt.show()
plt.savefig('drug-distribution.pdf')
