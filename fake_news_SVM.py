
import pandas as pd
import numpy as np
dataset=pd.read_csv('C:/Users/Intel/Desktop/fake.csv')
print("Types and counts of stories", dataset.groupby(["type"]).size())
all_news=dataset[dataset["type"]!='bs']
all_news.shape
fn=dataset[dataset['type']=='bs']
filtered_data=all_news[['title', 'text', 'type']]
filtered_data.title.fillna("", inplace=True)
filtered_data.text.fillna("", inplace=True)
filtered_data[0:2]
filtered_data['label']=filtered_data.type.map({
    'bias':1,
    'conspiracy':2,
    'hate':3,
    'satire':4,
    'state':5,
    'junksci':6,
    'fake':7
})
filtered_data.head(5)
all_text=filtered_data['title'] +" "+ filtered_data['text']
all_text.head(5)
x=all_text
x.shape
print(x.head(5))
y=filtered_data['label']
y.shape
x.head(5)
y.head(5)
from sklearn.feature_extraction.text import CountVectorizer
vect=CountVectorizer()
X_data=vect.fit_transform(x.values.astype('U')).todense()
X_data.shape
print(X_data)
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X_data,y,random_state=1,test_size=0.20)
print (X_train.shape)
print (Y_train.shape)
print (X_test.shape)
print (Y_test.shape)
from sklearn import svm
model=svm.SVC(kernel='linear')
model.fit(X_train,Y_train)
pred = model.predict(X_test)
from sklearn.metrics import accuracy_score
print("Accuracy Score : ",accuracy_score(pred,Y_test)*100)
import scikitplot as skplt
true_label=Y_test.map({
    1:'bias',
    2:'conspiracy',
    3:'hate',
    4:'satire',
    5:'state',
    6:'junksci',
    7:'fake'
})
pred_label=pd.Series(pred).map({
    1:'bias',
    2:'conspiracy',
    3:'hate',
    4:'satire',
    5:'state',
    6:'junksci',
    7:'fake'
})
labels=['bias','conspiracy','hate','satire','state','junksci','fake']

from sklearn.metrics import confusion_matrix as cm
import matplotlib.pyplot as plt  
c=cm(true_label,pred_label,labels)
print(c)
fig=plt.figure()
ax=fig.add_subplot(111)
cax=ax.matshow(c)
ax.set_xticklabels(['']+labels)
ax.set_yticklabels(['']+labels)

skplt.metrics.plot_confusion_matrix(true_label,pred_label,labels=labels, normalize=False, x_tick_rotation=45)
from sklearn.metrics import classification_report
print(classification_report(Y_test,pred))
