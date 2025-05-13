# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import dataset and print head,info of the dataset
2. Check for null values
3. Import kmeans and fit it to the dataset
4. Plot the graph using elbow method
5. Print the predicted array
6. Plot the customer segments 

## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: Aman Singh
RegisterNumber: 212224040020
*/
```
```python
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
df= pd.read_csv("/content/Mall_Customers.csv")


df.head()
df.info()
df.isnull().sum()

X = df[['Annual Income (k$)', 'Spending Score (1-100)']]
from sklearn.cluster import KMeans
wcss=[]
for i in range(1,11):
  kmeans=KMeans(n_clusters=i,init="k-means++",n_init=10)
  kmeans.fit(df.iloc[:,3:])
  wcss.append(kmeans.inertia_)


import matplotlib.pyplot as plt
plt.plot(range(1,11),wcss)
plt.xlabel("No of clusters")
plt.ylabel("wcss")
plt.title("Elbow method")
km=KMeans(n_clusters=5,n_init=10)

km.fit(df.iloc[:,3:])
y_pred=km.predict(df.iloc[:,3:])
y_pred


df["cluster"]=y_pred
dt0=df[df["cluster"]==0]
dt1=df[df["cluster"]==1]
dt2=df[df["cluster"]==2]
dt3=df[df["cluster"]==3]
dt4=df[df["cluster"]==4]
plt.scatter(dt0["Annual Income (k$)"],dt0["Spending Score (1-100)"],c="red",label="cluster1")
plt.scatter(dt1["Annual Income (k$)"],dt1["Spending Score (1-100)"],c="black",label="cluster2")
plt.scatter(dt2["Annual Income (k$)"],dt2["Spending Score (1-100)"],c="blue",label="cluster3")
plt.scatter(dt3["Annual Income (k$)"],dt3["Spending Score (1-100)"],c="green",label="cluster4")
plt.scatter(dt4["Annual Income (k$)"],dt4["Spending Score (1-100)"],c="magenta",label="cluster5")
plt.legend()
plt.title("Customer Segments")
```
## Output:
<img width="619" alt="Screenshot 2025-05-13 at 11 47 27 AM" src="https://github.com/user-attachments/assets/2c01e37c-c974-49e0-8271-487d7d55011c" />
<br>
<img width="474" alt="Screenshot 2025-05-13 at 11 47 42 AM" src="https://github.com/user-attachments/assets/35af1277-2518-4fc1-8e4b-ccb8857d71cc" />
<br>
<img width="376" alt="Screenshot 2025-05-13 at 11 47 52 AM" src="https://github.com/user-attachments/assets/a8b651e1-92e9-47fd-a177-1b11fbf93856" />
<br>
<img width="687" alt="Screenshot 2025-05-13 at 11 48 07 AM" src="https://github.com/user-attachments/assets/43387121-d1e2-4a8b-85cd-2771aaf5b6d1" />
<br>
<img width="597" alt="Screenshot 2025-05-13 at 12 46 21 PM" src="https://github.com/user-attachments/assets/9d7d861a-696e-471f-9f49-649428bc2a32" />
<br>
<img width="585" alt="Screenshot 2025-05-13 at 12 46 38 PM" src="https://github.com/user-attachments/assets/2508e948-030d-4063-a759-4836491dac91" />
<br>

## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
