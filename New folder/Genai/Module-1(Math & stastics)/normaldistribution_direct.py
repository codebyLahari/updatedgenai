import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data=np.random.normal(20,4,100)

plt.figure(figsize=(10, 6))
sns.histplot(data, bins=15, kde=True, color='blue')
plt.title('Histogram of Normally Distributed Data')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
print(data)
