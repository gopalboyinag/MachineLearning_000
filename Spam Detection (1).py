
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np


# In[44]:


messages=pd.read_csv('EmailSpam.txt',sep='\t', names=["label","message"])


# In[45]:


messages.head()


# In[46]:


messages.groupby('label').describe()


# In[48]:


messages['len']=messages['message'].apply(len)


# In[50]:


messages.head()


# In[51]:


import seaborn as sns


# In[54]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[62]:


sns.distplot(messages['len'], kde=True,hist=True,bins=20)


# In[63]:


messages.hist(column='len',by='label',bins=20)


# In[64]:


sns.pairplot(data=messages)


# In[70]:


import string


# In[71]:


mess="Selenim ! @ Demo automation machine leanring & Hello ,"


# In[79]:


nopunc=[char for char in mess if char not in string.punctuation]

nopunc=''.join(nopunc)

nopunc


# In[85]:


import nltk


# In[95]:


nltk.download('stopwords')


# In[96]:


from nltk.corpus import stopwords


# In[97]:


stopwords.words('english')[0:10]


# In[108]:


clean_mess=[word for word in nopunc.split() if word.lower() not in stopwords.words('english')]


# In[109]:


clean_mess


# In[110]:


def test_process(mess):
    nopunc=[char for char in mess if char not in string.punctuation]
    nopunc=''.join(nopunc)
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]


# In[139]:


mes=messages['message'].apply(test_process)


# In[119]:


from sklearn.feature_extraction.text import CountVectorizer


# In[120]:


bow_transformer=CountVectorizer(analyzer=test_process).fit(messages['message'])


# In[122]:


len(bow_transformer.vocabulary_)


# In[123]:


messages4=messages['message'][3]


# In[124]:


messages4


# In[126]:


bow4=bow_transformer.transform([messages4])


# In[128]:


print(bow4)


# In[131]:


print(bow_transformer.get_feature_names()[9554])
print(bow_transformer.get_feature_names()[4068])


# In[137]:


messages_bow=bow_transformer.transform(messages['message'])


# In[141]:


messages_bow.shape


# In[144]:


from sklearn.feature_extraction.text import TfidfTransformer

tfidf_transformer =TfidfTransformer().fit(messages_bow)

ifidf4=tfidf_transformer.transform(bow4)
print(ifidf4)


# In[145]:


messages_tfid=tfidf_transformer.transform(messages_bow)


# In[146]:


messages_tfid.shape


# In[149]:


from sklearn.naive_bayes import MultinomialNB
spam_detect_model=MultinomialNB().fit(messages_tfid,messages['label'])


# In[152]:


print('pridicted', spam_detect_model.predict(ifidf4)[0])
print('expectd:', messages.label[3])

