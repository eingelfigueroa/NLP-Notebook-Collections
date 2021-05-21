#!/usr/bin/env python
# coding: utf-8

# In[17]:


import pickle
import random
import spacy
nlp = spacy.load('en_core_web_sm')

with open('aspectz.pkl', 'rb') as f:
    aspect = pickle.load(f)
    
print(aspect['I can easily log-in and log-out my Canvas account. '])
aspect['I can easily log-in and log-out my Canvas account. ']


# In[6]:


def unique_list(l):
    ulist = []
    [ulist.append(x) for x in l if x not in ulist]
    return ulist


# In[5]:


sentiment = ["postiive", "neutral", "negative"]

sentiment = 'neutral'
tok = ''
for ap in aspect.values():
    
    if len(str(ap)) != 0:
        doc = nlp(str(ap))

        for token in doc:
            if token.pos_ == 'NOUN':
                tok += token.text +' ' 
                #print(token.text)
            if token.pos == 'ADJ':
                print(token)
            if token.pos == 'VERB':
                print(token)

answer =' '.join(unique_list(tok.split()))

if sentiment == 'postive':
    print("The students enjoyed the services of " + a)
elif sentiment == 'neutral':
    print("There is no immediate action needed for " + a +" but needs improvment")
elif sentiment == 'negative':
    print("There is significant unsatisfaction in these terms "+ a + "provide immediate intervention")
    


# In[27]:


tryme = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 10: [], 11: [], 12: ['logging in'], 13: [], 14: [], 15: [], 16: ['easily log'], 17: [], 18: [], 19: [], 20: ['have out'], 21: [], 22: [], 23: [], 24: [], 25: ['easily log', 'faster internet'], 26: [], 27: [], 28: [], 29: [], 30: [], 31: ['easily log'], 32: [], 33: ['log easily'], 34: [], 35: ['easily log'], 36: [], 37: [], 38: [], 39: ['easily access'], 40: [], 41: ['easily login', 'slow to open'], 42: ['easily modify'], 43: [], 44: [], 45: ['easily log', 'Strongly Agree'], 46: [], 47: ['log anytime', 'access anywhere', 'easily access'], 48: [], 49: ['still encounter'], 50: [], 51: ['strongly agree', 'easily access'], 52: [], 53: [], 54: [], 55: [], 56: ['easily log', "always 'm"], 57: ['easily log'], 58: [], 59: ['working properly'], 60: ['easily log'], 61: ['easily logi'], 62: ['easily log'], 63: ['easily log'], 64: ['easily log'], 65: [], 66: ['easily access'], 67: ['easily access'], 68: ['encountered far'], 69: [], 70: ['Strongly Agree'], 71: ['operated quickly', 'logging in'], 72: ['access easily'], 73: ['Strongly agree'], 74: ['log ever'], 75: ['easily log'], 76: ['yes easy to logout'], 77: ['Fast process'], 78: [], 79: [], 80: [], 81: ['strongly agree'], 82: ['Strongly Agree'], 83: [], 84: [], 85: ['Strongly Agree'], 86: [], 87: ['access easily'], 88: [], 89: [], 90: ['log easily'], 91: ['access easily'], 92: ['easily login'], 93: [], 94: [], 95: ['log anywhere', 'easily login'], 96: ['easily log'], 97: ['easily login'], 98: [], 99: ['easily log'], 100: [], 101: ['easily log', 'smooth computer'], 102: ['easily log', 'always log'], 103: [], 104: [], 105: ['easily log', 'fast internet connection'], 106: [], 107: [], 108: [], 109: [], 110: [], 111: [], 112: ['easily log'], 113: ['make fast'], 114: [], 115: [], 116: [], 117: [], 118: ['easily log'], 119: [], 120: ['easily log'], 121: ['other browser'], 122: [], 123: ['easily login'], 124: ['NEEDS ALSO', 'TOTALLY LOG'], 125: [], 126: [], 127: [], 128: ['easy log'], 129: [], 130: ['log easily'], 131: [], 132: [], 133: [], 134: [], 135: [], 136: ['strongly agree'], 137: [], 138: ['other computers'], 139: []}


# In[24]:


def actionPlan(aspect,sentiment):
    tok = ''
    for ap in aspect.values():

        if len(str(ap)) != 0:
            doc = nlp(str(ap))

            for token in doc:
                if token.pos_ == 'NOUN':
                    tok += token.text +' ' 
                    #print(token.text)
                if token.pos == 'ADJ':
                    print(token)
                if token.pos == 'VERB':
                    print(token)

    answer =' '.join(unique_list(tok.split()))

    if sentiment == 'postive':
        return "The students enjoyed the services of " + answer
    elif sentiment == 'neutral':
        return "There is no immediate action needed for " + answer +" but needs improvment"
    else:
        return "There is significant unsatisfaction in terms of ["+ answer + "] provide immediate intervention"


# In[25]:


asp = aspect['I can easily log-in and log-out my Canvas account. ']
sentiment = random.choice(["postiive", "neutral", "negative"])

asp


# In[33]:


actionPlan(asp,sentiment)


# In[62]:


import pandas as pd
with open('sentiment.pkl', 'rb') as f:
    sentiment = pickle.load(f)
sent = sentiment.values()
score = []
for i in sent:
    for j in i.values():
        score.append(j)


# In[68]:


sent = sentiment.keys()
key = []
for i in sent:
    key.append(i)


# In[46]:


def f(row):
    if row['Sentiment Score'] >= 0.5:
        val = 'Positive'
    elif row['Sentiment Score'] <= -0.5:
        val = 'Negative'
    else:
        val = 'Neutral'
    return val


# In[160]:


data = {
    'Title':  key,
    'Sentiment Score':  comp
}


# In[161]:


df = pd.DataFrame(data)
df.to_csv('new_data.csv',index=False)


# In[165]:


import numpy as np

conditionlist = [
    (df['Sentiment Score'] >= 0.05) ,
    (df['Sentiment Score'] > 0.03) & (df['Sentiment Score'] < 0.05),
    (df['Sentiment Score'] <= 0.03)]
choicelist = ['Positive', 'Neutral', 'Negative']
df['Sentiment'] = np.select(conditionlist, choicelist, default='Not Specified')


# In[166]:


df.to_csv('new_data_sentiment.csv',index=False)


# In[168]:


df


# In[14]:


from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np
import nltk
nltk.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger') 
nltk.download('punkt')
nltk.download('wordnet')
import re
from nltk import word_tokenize, pos_tag, pos_tag_sents
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from nltk.stem import WordNetLemmatizer 
from nltk.corpus import wordnet 
# Create WordNetLemmatizer object 

sid = SentimentIntensityAnalyzer()
lemmatizer = WordNetLemmatizer() 

stoplist = stopwords.words('english') + ['though']


# In[15]:


def remove_stopWords(w): 
    stoplist = stopwords.words('english') + ['though']
    w = ' '.join(word for word in w.split() if word not in stoplist)
    return w
def pos_tagger(nltk_tag): 
    if nltk_tag.startswith('J'): 
        return wordnet.ADJ 
    elif nltk_tag.startswith('V'): 
        return wordnet.VERB 
    elif nltk_tag.startswith('N'): 
        return wordnet.NOUN 
    elif nltk_tag.startswith('R'): 
        return wordnet.ADV 
    else:           
        return None


# In[16]:



static_df = pd.read_csv('opinion_survey.csv')
stoplist = stopwords.words('english') + ['though']
c_vec = CountVectorizer(stop_words=stoplist, ngram_range=(3,5))

sid = SentimentIntensityAnalyzer()
lemmatizer = WordNetLemmatizer() 

df = pd.read_csv('opinion_survey.csv')

df = df.replace(np.nan, 'Neutral', regex=True)

col_range = len(df.columns) # number of columns
stored_pos = []
for i in range(0,col_range):
    col = df.columns[i] # The current column
    df.loc[:,col] = df[col].apply(lambda x : str.lower(str(x))) ## To Lower Case
    df.loc[:,col] = df[col].apply(lambda x : " ".join(re.findall('[\w]+',x))) # Remove Punctuations
    df.loc[:,col] = df[col].apply(lambda x : remove_stopWords(x)) # Remove Stop words

    ##POS TAGGING
    texts = df.loc[:,col].tolist()
    tagged_texts = pos_tag_sents(map(word_tokenize, texts)) ### Tag every word in a row with POS

    ### Lemmatization
    new = []

    stored_pos.append(tagged_texts)
    for i in tagged_texts:
        #if len(i) > 0:
        lemmatized_sentence = []
        for word, tag in i:
            tag = pos_tagger(tag) ### Convert POS Tag to known POS for simplification
            if tag is None: 
    # if there is no available tag, append the token as is 
                lemmatized_sentence.append(word) 
            else:         
    # else use the tag to lemmatize the token 
                lemmatized_sentence.append(lemmatizer.lemmatize(word, tag)) 

        lemmatized_sentence = " ".join(lemmatized_sentence) 
        #print(lemmatized_sentence)
        new.append(lemmatized_sentence)

    else:
        pass


    df['POS'] = new ## Store tagged words


df = df.replace(r'^\s*$', "neutral", regex=True) ## If row value is null, replace with neutral string
df = df.iloc[:,:-1]

static_df = static_df.replace(r'^\s*$', "neutral", regex=True)
static_df = static_df.replace(np.nan, 'Neutral', regex=True)
static_df


# In[17]:


df


# In[18]:



comp = []
col_range = len(df.columns) # number of columns

for i in range(0,col_range):
    col = df.columns[i] # The current column
    ngrams = c_vec.fit_transform(static_df[col])
    vocab = c_vec.vocabulary_
    vocab = vocab.keys()

    df = pd.DataFrame(vocab, columns = ['ngram'])
    
    df['scores'] = df['ngram'].apply(lambda x: sid.polarity_scores(x)) ## Get polarity score of every Column
    compound = df['scores'].apply(lambda score_dict: score_dict['compound']) ## Extract the compound from the results
    df = df.drop('scores', 1) # Drop score DF in every iteration

    ave = np.average(compound)# Get the mean compound of each columns
    #print(ave)
    comp.append(ave)

print(comp)


# In[19]:


ndf = pd.DataFrame(data = np.array([comp]), columns=df.columns)
ndf = ndf.to_dict()


# In[20]:


from sklearn.feature_extraction.text import CountVectorizer
from pprint import pprint

c_vec = CountVectorizer(stop_words=stoplist, ngram_range=(3,5))
comp = []
col_range = len(static_df.columns) # number of columns

for i in range(0,col_range):
    #col = test_df.columns[i] # The Unlemmatized Opinion
    col = static_df.columns[i] # The Lemmatized Opinion
    
    ngrams = c_vec.fit_transform(static_df[col])
    vocab = c_vec.vocabulary_
    vocab = vocab.keys()
    
    #Create dataframe to store different lengths of ngrams
    ## Note Cannot properly store Ngrams due to uneven row count, so every iteration the previous column is dropped
    df = pd.DataFrame(vocab, columns = ['ngram'])
    
    df['scores'] = df['ngram'].apply(lambda x: sid.polarity_scores(x)) ## Get polarity score of every Column and Row
    compound = df['scores'].apply(lambda score_dict: score_dict['compound']) ## Extract the compound from the results
    
    ave = np.average(compound)# Get the mean compound of each columns
    comp.append(ave) # Save mean and append to list
    #print("Length of Ngram in column:",i+1,len(vocab))

    
        
    #print("Sentiment Score per row: ",comp) ##
    print("Mean Sentiment Score for Column",i+1,":",ave) ## 


# In[157]:


labels = static_df.columns
labels = list(labels)

new_sent = [dict(zip(labels, datum)) for datum in [comp]]


# In[21]:


comp


# In[80]:


nlp = spacy.load('en_core_web_sm')


# In[158]:


with open('sentiment.pkl', 'wb') as f:
    sent = pickle.dump(new_sent[0], f)


# In[24]:


import pickle
import pandas as pd

with open('sentiment.pkl', 'rb') as f:
    sent = pickle.load(f)

with open('aspect.pkl', 'rb') as f:
    aspect = pickle.load(f)


# In[126]:


df = pd.DataFrame(list(aspect.items()),columns=['Question','Aspect'])
asp_acad = [findAc(acad_filter,i) for i in df['Aspect']]
df.insert(loc=2, column='Sentiment_Score', value=comp)

df.insert(loc=2, column='Acad_aspect', value=asp_acad)


# In[127]:


def unique_list(l):
    ulist = []
    [ulist.append(x) for x in l if x not in ulist]
    return ulist

def parse_values(x):
    if x>= 0.05 :
        return 'POSITIVE'
    elif x<= 0.03:
        return 'NEGATIVE'
    else:
        return 'NEUTRAL'
    
def parse_actionPlan(df):
    tok = ''
    for ap in (df['Acad aspect']):

        if len(str(ap)) != 0:
            doc = nlp(str(ap))

            for token in doc:
                if token.pos_ == 'NOUN':
                    tok += token.text +' ' 
                    #print(token.text)
                if token.pos == 'ADJ':
                    print(token)
                if token.pos == 'VERB':
                    print(token)

    answer =' '.join(unique_list(tok.split()))
    return answer


# In[128]:


for i in range(0, 21):

    df['Sentiments'] = df['Sentiment_Score'].apply(parse_values)
df


# In[129]:


def plan(sentiment,answer):
    if sentiment == 'POSITIVE':
        return "The students enjoyed the services of: " + answer
    elif sentiment == 'NEGATIVE':
        return "There is significant unsatisfaction in terms of "+ answer + " provide immediate intervention" 
    elif sentiment == 'NEUTRAL':
        return "There is no immediate action needed for " + answer +" but needs improvement"
    else:
        return "No Aspect and Comment to decide on"


# In[130]:


nndf = df[df.astype(str)['Acad_aspect'] != '[]']

for i in range(0, len(ndf['Question'])):
    nndf['Action_Plan'] = df.apply(lambda row: plan(row.Sentiments,str(row.Acad_aspect)), axis=1)   
nndf


# In[7]:


import spacy
nlp = spacy.load('en_core_web_sm')


# In[10]:


def findAc(filtr, word):
    r = re.compile('|'.join([r'\b%s\b' % w for w in filtr]), flags=re.I)
    results = r.findall(str(word))
    
    return results

def actionPlan(aspect,sentiment):
    tok = ''
    for ap in aspect.values():

        if len(str(ap)) != 0:
            doc = nlp(str(ap))

            for token in doc:
                if token.pos_ == 'NOUN':
                    tok += token.text +' ' 
                    #print(token.text)
                if token.pos == 'ADJ':
                    print(token)
                if token.pos == 'VERB':
                    print(token)

    answer =' '.join(unique_list(tok.split()))

    if sentiment >= 0.2 and len(answer) != 0:
        return "The students enjoyed the services of: " + answer
    elif sentiment <= 0.15 and len(answer) != 0:
        return "There is significant unsatisfaction in terms of ["+ answer + "] provide immediate intervention" 
    elif sentiment > 0.15 and sentiment < 0.2 and len(answer) != 0:
        return "There is no immediate action needed for [" + answer +"] but needs improvement"
    else:
        return "No Aspect and Comment to decide on"


# In[32]:


acad_filter = ['subject','teacher','teach','faculty',
                'professor','school','system','learning',
                'modules','module','teaching'
                'assignments','assignment','knowledge','activities']

ito_filter = ['system','internet','connection',
                'slow','laboratory','access',
                'equipment']

itbl_filter = ['canvas','design','slow','platform',
                'application','survey',
                'modules','modules','log']


# In[ ]:




