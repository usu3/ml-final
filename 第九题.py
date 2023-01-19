#%%
#第9题代码
import os
import sys
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.classify import NaiveBayesClassifier
from random import shuffle
from nltk.classify.util import accuracy
def load_from_dir(path):
    files_name=os.listdir(path)
    txt_lyst=[]
    for name in files_name:
         with open(os.path.join(path,name),'r',encoding='latin-1') as f:
            txt_lyst.append(f.read())
    return txt_lyst
def load_files():
    ham=[]
    spam=[]
    path='./pre-processed form/'
    for i in range(6):
        ham+=load_from_dir(os.path.join(path,f'enron{i+1}/ham'))
        spam+=load_from_dir(os.path.join(path,f'enron{i+1}/spam'))
    return ham,spam
def feature_extraction(txt,stopwords=[]):
    featureset={}
    for word in txt:
        if word not in stopwords and word not in featureset:
            featureset[word]=True
    return featureset
if __name__ == '__main__':
    ham,spam=load_files()
    dot=', . < > / ? ; : [ ] { } | \' \" ( ) - _ = + ` ~ ! @ # $ % ^ & *'.split()
    ham_dataset=[]
    spam_dataset=[]
    for mail in ham:
        word_list=word_tokenize(mail.lower())
        word_set=feature_extraction(word_list,stopwords.words('english')+dot)
        ham_dataset.append((word_set,'ham'))
    for mail in spam:
        word_list=word_tokenize(mail.lower())
        word_set=feature_extraction(word_list,stopwords.words('english')+dot)
        spam_dataset.append((word_set,'spam')) 
    shuffle(ham_dataset)
    shuffle(spam_dataset)
    train,test=ham_dataset[:int(0.8*len(ham_dataset))]+spam_dataset[:int(0.8*len(ham_dataset))],\
        ham_dataset[int(0.8*len(ham_dataset)):]+spam_dataset[int(0.8*len(ham_dataset)):]
    classifier = NaiveBayesClassifier.train(train)
    acc=accuracy(classifier,test)       

