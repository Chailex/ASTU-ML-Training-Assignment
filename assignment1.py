import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

#Loading the dataset
pcaData = pd.read_csv("PCA_practice_dataset.csv")
X=pcaData.to_numpy()
print(X.shape)

scaler = StandardScaler()
X = scaler.fit_transform(X)

#PCA
pca = PCA()
X = pca.fit_transform(X)

cumalitiveVarience = np.cumsum(pca.explained_variance_ratio_)*100
thresholds = [i for i in range(90,98)]

components = [np.argmax(cumalitiveVarience>threshold) for threshold in thresholds]

for component, threshold in zip(components, thresholds):
    print("Component for threshold"+str(threshold)+"% is :",component)
      

#dimensionality reduction
A=X
for component, threshold in zip(components, thresholds):
    pca=PCA(n_components=component)
    newX=pca.fit_transform(A)
    print("\nThreshold: ", threshold)
    print("Shape after dimensionalty reduction: ", newX.shape)
    
#Plotting the data on scree plot
plt.ylabel("Threshold")
plt.xlabel("Principle component")
plt.plot(components,range(90,98),'o-') 

