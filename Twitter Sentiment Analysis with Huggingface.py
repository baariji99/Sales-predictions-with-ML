import  numpy as np
import pandas as pd
from tqdm import tqdm
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd
from tqdm import tqdm
tqdm.pandas()  # This line enables the progress_apply method

import nltk
nltk.download('stopwords')
print(stopwords.words('english'))
columnn_names =['target','id','date','flag','user','text']
#df = pd.read_csv("~/Downloads/training.1600000.processed.noemoticon.csv")
twitter_data = pd.read_csv('~/Downloads/training.1600000.processed.noemoticon.csv', encoding='ISO-8859-1',names=columnn_names)
print(twitter_data.isnull().sum())
print(twitter_data['target'].value_counts())
twitter_data['target'].replace({'1':'4'} , inplace=True)

#Stemming
port_stem=PorterStemmer()
def stemming(content):
    stemmed_content=re.sub('[^a-zA-Z]',' ',content)
    stemmed_content=stemmed_content.lower()
    stemmed_content=stemmed_content.split()
    stemmed_content=[port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content=' '.join(stemmed_content)
    return  stemmed_content
tqdm.pandas()
twitter_data['stemmed_content']=twitter_data['text'].apply(stemming)
x=twitter_data['stemmed_content']
y=twitter_data['target']
x_train ,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,stratify=y,random_state=2)



#converting textual data to numerical data
vectorizer =TfidfVectorizer()
x_train=vectorizer.fit_transform(x_train)
x_test=vectorizer.fit_transform(x_test)
print(x_train,x_test)

# training machine model using logistics regression

model=LogisticRegression(max_iter=1000)
model.fit(x_train,y_train)

x_train_pred=model.predict(x_train)
trainng_data_accuracy=accuracy_score(y_train,x_train_pred)
print(trainng_data_accuracy,'accuracy on training data')

# accuracy score on test data
x_test_pred=model.predict(x_test)
test_data_accracy=accuracy_score(y_test,x_test_pred)
print("accuracy on test data",test_data_accracy)
import pickle
file_name="trained_model.sav"
pickle.dump(model,open(file_name,'wb'))
# using the saved model for future prediction




