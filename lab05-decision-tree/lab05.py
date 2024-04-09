#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import pickle


# ## 5.2 Data preparation

# In[2]:


from sklearn import datasets

data_breast_cancer = datasets.load_breast_cancer(as_frame=True)
data_breast_cancer.data.head()


# In[3]:


size = 300
X = np.random.rand(size)*5-2.5
w4, w3, w2, w1, w0 = 1, 2, 1, -4, 2
y = w4*(X**4) + w3*(X**3) + w2*(X**2) + w1*X + w0 + np.random.randn(size)*8-4

df = pd.DataFrame({'x':X, 'y':y})
df.plot.scatter(x='x', y='y')


# ## 5.3 Classification

# In[4]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.tree import export_graphviz
import graphviz

X_train, X_test, y_train, y_test = train_test_split(
    data_breast_cancer['data'][['mean texture', 'mean symmetry']],
    data_breast_cancer['target'].astype(np.uint8), 
    test_size=0.2,
    random_state=42
)

results = []
for depth in range(1, 10):
    tree_clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
    tree_clf.fit(X_train, y_train)

    y_pred_train = tree_clf.predict(X_train)
    y_pred_test = tree_clf.predict(X_test)
    
    f1_train = f1_score(y_train, y_pred_train)
    f1_test = f1_score(y_test, y_pred_test)
    
    acc_train = accuracy_score(y_train, y_pred_train)
    acc_test = accuracy_score(y_test, y_pred_test)
    
    results.append([depth, f1_train, f1_test, acc_train, acc_test])
    print(f"depth: {depth}, f1_train: {f1_train}, f1_test: {f1_test}")

best_result = max(results, key=lambda x: x[2])
best_depth = best_result[0]
print(f"\nbest result at depth: {best_depth} {best_result}")

with open('f1acc_tree.pkl', 'wb') as f:
    pickle.dump(best_result, f)

clf = DecisionTreeClassifier(max_depth=best_depth, random_state=42)
clf.fit(X_train, y_train)

dot_data = export_graphviz(
    clf, 
    out_file=None, 
    feature_names=['mean texture', 'mean symmetry'], 
    class_names=['malignant', 'benign'], 
    rounded=True, 
    filled=True
)

graph = graphviz.Source(dot_data)  

with open('bc.png', 'wb') as f:
    f.write(graph.pipe(format='png'))

graph


# ## 5.4 Regression

# In[5]:


from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = X_train.reshape(-1, 1)
X_test = X_test.reshape(-1, 1)

results=[]
for depth in range(1, 10):
    tree_reg = DecisionTreeRegressor(max_depth=depth, random_state=42)
    tree_reg.fit(X_train, y_train)
    
    train_pred = tree_reg.predict(X_train)
    test_pred = tree_reg.predict(X_test)
    
    mse_train = mean_squared_error(y_train, train_pred)
    mse_test = mean_squared_error(y_test, test_pred)
    
    results.append([depth, mse_train, mse_test])

best_result = min(results, key=lambda x: x[2])
print(best_result)

with open('mse_tree.pkl', 'wb') as f:
    pickle.dump(best_result, f)

reg = DecisionTreeRegressor(max_depth=best_depth, random_state=42)
reg.fit(X_train, y_train)

dot_data = export_graphviz(
    reg, 
    out_file=None,
    rounded=True, 
    filled=True
)

graph = graphviz.Source(dot_data)  

with open('reg.png', 'wb') as f:
    f.write(graph.pipe(format='png'))

graph


# In[6]:


y_pred = reg.predict(X_test)

plt.figure(figsize=(10, 6))
plt.scatter(df['x'], df['y'], label='Data')
plt.plot(X_test, y_pred, color='red', label='Regressor prediction')

plt.legend()
plt.show()


# In[ ]:




