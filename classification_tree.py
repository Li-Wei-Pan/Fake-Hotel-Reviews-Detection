#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import sklearn
import joblib
import nltk
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import f1_score, precision_score, accuracy_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')


# In[2]:


train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
STOPWORDS = list(stopwords.words("english"))


# In[3]:


X_train, y_train = train_df["Combined_Review"], train_df["source"]
X_test,  y_test  = test_df["Combined_Review"],  test_df["source"]
cv = StratifiedKFold(n_splits= 5, shuffle= True, random_state= 42)


# In[4]:


vectorizer = CountVectorizer()


# **Single Classification Tree**

# In[5]:


ct_pipeline = Pipeline([
    ('vectorizer', CountVectorizer(
        min_df=2,
        max_df=0.95
    )),
    ('top_k', SelectKBest(chi2, k=1000)),
    ('ct', DecisionTreeClassifier(
        random_state=42
    ))
])

ct_pipeline.fit(X_train, y_train)


# In[7]:


param_grid_ct = {
    "ct__criterion": ['gini', 'entropy', 'log_loss'],
    'ct__splitter': ['best'],
    'ct__max_depth': [5, 10, 15, 20],
    'ct__min_samples_split': [2,5,10,15],
    'ct__ccp_alpha': [0.0, 0.0005, 0.001, 0.002, 0.003, 0.005, 0.0075, 0.01, 0.02],
    'ct__min_samples_leaf': [1,2,4,8],
    'ct__max_features': [None],
    'vectorizer__stop_words': [None, STOPWORDS],
}

grid_ct = GridSearchCV(
    ct_pipeline,
    param_grid_ct,
    cv=cv,
    scoring='accuracy',
    n_jobs=-1
)

grid_ct.fit(X_train, y_train)


# In[8]:


print(grid_ct.best_params_)
best_ct = grid_ct.best_estimator_


# In[9]:


y_pred_ct = best_ct.predict(X_test)
f1_ct = f1_score(y_test, y_pred_ct)
acc_ct = accuracy_score(y_test, y_pred_ct)
recall_ct = recall_score(y_test, y_pred_ct)
pre_ct = precision_score(y_test, y_pred_ct)
print('ct result')
print('f1',f1_ct)
print('-----')
print('accuracy: ', acc_ct)
print('-----')
print('recall:', recall_ct)
print('-----')
print('precision: ', pre_ct)
print('-----')

pd.DataFrame(y_pred_ct).to_csv("data/y_pred_ct.csv", index=False)


# In[10]:


vec = best_ct.named_steps['vectorizer']
sel = best_ct.named_steps['top_k']      
tree = best_ct.named_steps['ct']

names = vec.get_feature_names_out()[sel.get_support()]
imp = tree.feature_importances_
top = np.argsort(imp)[::-1][:5]

for n, v in zip(names[top], imp[top]):
    print(f"{n}: {v:.4f}")


# In[11]:


joblib.dump(best_ct, 'models/ct.joblib')

