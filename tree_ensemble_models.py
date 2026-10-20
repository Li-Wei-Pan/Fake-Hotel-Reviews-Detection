#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
import pandas as pd
import sklearn
import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import f1_score, precision_score, accuracy_score, recall_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


# In[2]:


# load data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')


# In[ ]:


X_train, y_train = train_df["Combined_Review"], train_df["source"]
X_test,  y_test  = test_df["Combined_Review"],  test_df["source"]
cv = StratifiedKFold(n_splits= 5, shuffle= True, random_state= 42)


# In[ ]:


vectorizer = CountVectorizer()


# **RANDOM FOREST CLASSIFIER**

# In[11]:


rf_pipeline = Pipeline([
    ('vectorizer', CountVectorizer(
        min_df=2,
        max_df=0.95
    )),
    ('top_k', SelectKBest(chi2, k=1000)),
    ('rf', RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ))
])

rf_pipeline.fit(X_train, y_train)


# In[17]:


param_grid_rf = {
    'vectorizer__min_df': [1, 2, 5],
    'vectorizer__ngram_range': [(1, 1), (1,2)],
    'top_k__k': [500, 1000, 2000, 2500],

    'rf__n_estimators': [200, 500, 750],
    'rf__max_depth': [None, 20],
    'rf__min_samples_leaf': [1, 2],
    'rf__max_features': ['sqrt']
}

grid_rf = GridSearchCV(
    rf_pipeline,
    param_grid_rf,
    cv=cv,
    scoring='accuracy',
    n_jobs=-1
)

grid_rf.fit(X_train, y_train)


# In[20]:


print(grid_rf.best_params_)
# {'rf__max_depth': 20, 
# 'rf__max_features': 'sqrt', 
# 'rf__min_samples_leaf': 1, 
# 'rf__n_estimators': 500, 
# 'top_k__k': 2000, 
# 'vectorizer__min_df': 1, 
# 'vectorizer__ngram_range': (1, 1)}

#{'rf__max_depth': 20, 
# 'rf__max_features': 'sqrt',
#  'rf__min_samples_leaf': 1, 
# 'rf__n_estimators': 500,
#  'top_k__k': 2000,
#  'vectorizer__min_df': 1, 
# 'vectorizer__ngram_range': (1, 1)}
best_rf = grid_rf.best_estimator_


# In[21]:


y_pred_rf = best_rf.predict(X_test)
f1_rf = f1_score(y_test, y_pred_rf)
acc_rf = accuracy_score(y_test, y_pred_rf)
recall_rf = recall_score(y_test, y_pred_rf)
pre_rf= precision_score(y_test, y_pred_rf)
print('rf result')
print('f1',f1_rf)
print('-----')
print('accuracy: ', acc_rf)
print('-----')
print('recall:', recall_rf)
print('-----')
print('precision: ', pre_rf)
print('-----')


# **GRADIENT BOOSTING CLASSIFIER**
# 

# In[27]:


gb_pipeline = Pipeline([
    ('vectorizer', CountVectorizer(
        min_df=2,
        max_df=0.95
    )),
    ('top_k', SelectKBest(chi2, k=1000)),
    ('gb', GradientBoostingClassifier(
        n_estimators=100,
        random_state=42
    ))
])

gb_pipeline.fit(X_train, y_train)


# In[28]:


param_grid_gb = {
    'vectorizer__min_df': [1, 2, 5],
    'vectorizer__ngram_range': [(1, 1), (1, 2)],
    'top_k__k': [500, 1000, 2000],

    'gb__n_estimators': [50, 100, 200],
    'gb__learning_rate': [0.05, 0.1],
    'gb__max_depth': [2, 3],
    'gb__min_samples_leaf': [1, 5],
    'gb__subsample': [0.8, 1.0]
}

grid_gb = GridSearchCV(
    gb_pipeline,
    param_grid_gb,
    cv=cv,
    scoring='accuracy',
    n_jobs=-1
)

grid_gb.fit(X_train, y_train)


# In[ ]:


print(grid_gb.best_params_)

#{'gb__learning_rate': 0.1, 
# 'gb__max_depth': 3, 
# 'gb__min_samples_leaf': 5, 
# 'gb__n_estimators': 200, 
# 'gb__subsample': 0.8, 
# 'top_k__k': 500, 
# 'vectorizer__min_df': 5, 
# 'vectorizer__ngram_range': (1, 1)}

best_gb = grid_gb.best_estimator_


# In[30]:


y_pred_gb = best_gb.predict(X_test)
f1_gb = f1_score(y_test, y_pred_gb)
acc_gb = accuracy_score(y_test, y_pred_gb)
recall_gb = recall_score(y_test, y_pred_gb)
pre_gb= precision_score(y_test, y_pred_gb)
print('gb result')
print('f1',f1_gb)
print('-----')
print('accuracy: ', acc_gb)
print('-----')
print('recall:', recall_gb)
print('-----')
print('precision: ', pre_gb)
print('-----')

