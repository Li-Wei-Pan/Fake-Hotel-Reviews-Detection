#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
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


train_df[train_df["Combined_Review"].str.contains(r"\bnotch\b")]["Combined_Review"].head()


# In[5]:


vectorizer = CountVectorizer()


# In[6]:


lg_pipeline = Pipeline([
    ('vectorizer', CountVectorizer(
        min_df=2,
        max_df=0.95
    )),

    ('lg', LogisticRegression(
        l1_ratio = 1, 
        solver = 'liblinear',
        random_state=42,
    ))
])

lg_pipeline.fit(X_train, y_train)


# In[7]:


param_grid_lg = {
    'lg__C': [0.01, 0.1, 1, 10, 100],
    'vectorizer__stop_words': [None, STOPWORDS]
}

grid_lg = GridSearchCV(
    lg_pipeline,
    param_grid_lg,
    cv=cv,
    scoring='accuracy',
    n_jobs=-1
)

grid_lg.fit(X_train, y_train)


# In[8]:


print(grid_lg.best_params_)

best_lg = grid_lg.best_estimator_


# In[9]:


y_pred_lg = best_lg.predict(X_test)
f1_lg = f1_score(y_test, y_pred_lg)
acc_lg = accuracy_score(y_test, y_pred_lg)
recall_lg = recall_score(y_test, y_pred_lg)
pre_lg = precision_score(y_test, y_pred_lg)
print('lr result')
print('f1',f1_lg)
print('-----')
print('accuracy: ', acc_lg)
print('-----')
print('recall:', recall_lg)
print('-----')
print('precision: ', pre_lg)
print('-----')

# save y_pred_lr for stat test
y_pred_lg_df = pd.DataFrame(y_pred_lg)
y_pred_lg_df.to_csv("data/y_pred_lg_from_py.csv", index=False)


# In[10]:


# show 5 features for each class

res = pd.DataFrame(grid_lg.cv_results_)
res["param_vectorizer__stop_words"] = res["param_vectorizer__stop_words"].map(
    lambda s: "none" if s is None else "nltk")
res = res.drop(columns="params")   # also contains the full list
res.to_csv("models/lr_cv_results_from_py.csv", index=False)


# In[11]:


vec = best_lg.named_steps['vectorizer']
lr  = best_lg.named_steps['lg']
names = vec.get_feature_names_out()
coefs = lr.coef_[0]

order = np.argsort(coefs)
print("Toward class 1:", list(zip(names[order[-5:][::-1]], coefs[order[-5:][::-1]].round(3))))
print("Toward class 0:", list(zip(names[order[:5]], coefs[order[:5]].round(3))))
print("Non-zero coefficients:", np.count_nonzero(coefs), "of", len(coefs))


# In[12]:


joblib.dump(best_lg, 'models/lg_from_py.joblib')

