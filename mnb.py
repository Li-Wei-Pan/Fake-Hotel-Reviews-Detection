#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import sklearn
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import f1_score, precision_score, accuracy_score, recall_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')


# In[2]:


# load data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
STOPWORDS = list(stopwords.words("english"))


# In[3]:


train_df.shape


# In[4]:


X_train, y_train = train_df["Combined_Review"], train_df["source"]
X_test,  y_test  = test_df["Combined_Review"],  test_df["source"]
sentiment_test = test_df["Sentiment"]  #for q3 only
cv = StratifiedKFold(n_splits= 5, shuffle= True, random_state= 42)

# save X_train
X_train.to_csv("data/X_train.csv", index = False)


# In[5]:


X_train.shape


# In[6]:


# BoW testing
vectorizer = CountVectorizer()
bows = vectorizer.fit_transform(X_train, y_train)

feature_names = vectorizer.get_feature_names_out()
print(feature_names)


# In[7]:


# Multinomial naive Bayes
multi_nb = sklearn.naive_bayes.MultinomialNB()
multi_nb.fit(bows, y_train)


# In[8]:


multi_nb_pipeline = Pipeline(
    [('vectorizer', CountVectorizer()),
     ('top_k', SelectKBest(chi2)),
    ('mnb', sklearn.naive_bayes.MultinomialNB())
    ]
)

multi_nb_pipeline.fit(X_train, y_train)


# In[9]:


cross_val_score(multi_nb_pipeline, X_train, y_train, cv = 5)


# In[10]:


multi_nb_pipeline[:-1].get_feature_names_out()


# In[11]:


param_grid = {
    "vectorizer__min_df": [1, 2, 5],
    "top_k__k": [50, 100, 250, 500, 1000, 2000, "all"],
    "mnb__alpha": [0.01, 0.1, 0.5, 1.0],
    "vectorizer__stop_words": [None, STOPWORDS]
}


grid_search = GridSearchCV(multi_nb_pipeline, param_grid, cv = cv, scoring = 'accuracy', n_jobs= -1)

grid_search.fit(X_train, y_train)
print(grid_search.best_params_)
print(grid_search.best_score_)  



# In[12]:


grid_search.best_params_


# In[13]:


best_mnb = grid_search.best_estimator_
# or this : final_mnb = mnb_pipeline.set_params(**grid_search.best_params_) to use the best parsms


# In[14]:


y_pred_mnb = best_mnb.predict(X_test)
f1_mnb = f1_score(y_test, y_pred_mnb)
acc_mnb = accuracy_score(y_test, y_pred_mnb)
recall_mnb = recall_score(y_test, y_pred_mnb)
pre_mnb= precision_score(y_test, y_pred_mnb)
print('MNB result')
print('f1',f1_mnb)
print('-----')
print('accuracy: ', acc_mnb)
print('-----')
print('recall:', recall_mnb)
print('-----')
print('precision: ', pre_mnb)
print('-----')

# save y_pred_mnb for stat test
y_pred_mnb_df = pd.DataFrame(y_pred_mnb)
y_pred_mnb_df.to_csv("data/y_pred_mnb.csv", index=False)


# In[15]:


grid_search.cv_results_


# In[16]:


# save best grid search
res = pd.DataFrame(grid_search.cv_results_)
res["param_vectorizer__stop_words"] = res["param_vectorizer__stop_words"].map(
    lambda s: "none" if s is None else "nltk")
res = res.drop(columns="params")   # also contains the full list
res.to_csv("models/mnb_cv_results.csv", index=False)


# In[17]:


best_mnb[:-1].get_feature_names_out()   # 2k selected words




# In[18]:


joblib.dump(best_mnb, 'models/mnb.joblib')


# In[19]:


'the' in best_mnb.named_steps['vectorizer'].vocabulary_


# In[20]:


#load model
best_mnb = joblib.load('models/mnb.joblib')


# In[21]:


mnb   = best_mnb.named_steps['mnb']
names = best_mnb[:-1].get_feature_names_out()

log_ratio = mnb.feature_log_prob_[1] - mnb.feature_log_prob_[0]   # log P(w|1) / P(w|0)
order = log_ratio.argsort()

top_class1 = names[order[::-1][:5]]  
top_class0 = names[order[:5]]  


# In[22]:


print(top_class0)
print('----')
print(top_class1)


# In[ ]:




