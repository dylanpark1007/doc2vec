import pandas

def txt2csv(txtname,csvname1,csvname2):
    f = open(txtname, "r",encoding='utf-8')
    label = []
    sentences = []
    while True:
        line = f.readline()
        if line == '':
            break
        line_token = line.split()
        label.append(line_token[0])
        sentence = ' '.join(line_token[1::])
        sentences.append(sentence)
    f.close()

    dataframe1 = pandas.DataFrame(label)
    dataframe1.to_csv(csvname1,encoding='utf-8-sig')
    dataframe2 = pandas.DataFrame(sentences)
    dataframe2.to_csv(csvname2, encoding='utf-8-sig')


txt2csv(txtname="./data/data/stsa.fine.dev", csvname1='./data/phrase_label_dev.csv',csvname2='./data/phrase_sentences_dev.csv')
txt2csv(txtname="./data/data/stsa.fine.phrases.train", csvname1='./data/phrase_label_phrase.csv',csvname2='./data/phrase_sentences_phrase.csv')
txt2csv(txtname="./data/data/stsa.fine.test", csvname1='./data/phrase_label_test.csv',csvname2='./data/phrase_sentences_test.csv')
txt2csv(txtname="./data/data/stsa.fine.train", csvname1='./data/phrase_label_train.csv',csvname2='./data/phrase_sentences_train.csv')

txt2csv(txtname="./data/data/stsa.binary.phrases.train", csvname1='./data/phrase_label.csv',csvname2='./data/phrase_sentences.csv')
txt2csv(txtname="./data/stsa.binary (2).txt", csvname1='./data/label2.csv',csvname2='./data/sentences2.csv')
txt2csv(txtname="./data/stsa.binary (3).txt", csvname1='./data/label3.csv',csvname2='./data/sentences3.csv')

import pandas
import os
from os import listdir
from os.path import isfile, join
import re

def cleanText(readData):
    text = re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]', '', readData)
    return text

def takeSecond(elem):
    return elem[0]

def txt2csv_imdb(mypath):
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    doc = []
    label = []
    sentences = []
    for file in onlyfiles:
        address = os.path.join(mypath, file)
        filename1 = file.split('.')
        filename2 = filename1[0]
        filename3 = filename2.split('_')
        doc.append(int(filename3[0]))
        label.append(int(filename3[1]))
        with open(address, 'r',encoding='utf-8') as f:
            line = f.read()
            line = line.replace('-',' ')
            line = line.replace(',', ' ')
            line = line.replace('?', ' ')
            line = line.replace('!', ' ')
            line = line.replace('"', ' ')
            line = line.replace("'", " ")
            line_token = line.split()
            line_token = [cleanText(token).lower() for token in line_token]
            line_token = ' '.join(line_token)
            sentences.append(line_token)
    data = list(zip(doc, label, sentences))
    data.sort(key=takeSecond)
    doc, label, sentences = zip(*data)

    doc = list(doc)
    label = list(label)
    sentences = list(sentences)

    return label, sentences



mypath1 = 'data/aclImdb_v1/aclImdb/train/pos'
label1, sentences1 = txt2csv_imdb(mypath1)
mypath2 = 'data/aclImdb_v1/aclImdb/train/neg'
label2, sentences2 = txt2csv_imdb(mypath2)
mypath3 = 'data/aclImdb_v1/aclImdb/test/pos'
label3, sentences3 = txt2csv_imdb(mypath3)
mypath4 = 'data/aclImdb_v1/aclImdb/test/neg'
label4, sentences4 = txt2csv_imdb(mypath4)
mypath5 = 'data/aclImdb_v1/aclImdb/train/unsup'
label5, sentences5 = txt2csv_imdb(mypath5)


label = []
sentences = []

label.extend(label1)
label.extend(label2)
label.extend(label3)
label.extend(label4)
label.extend(label5)
sentences.extend(sentences1)
sentences.extend(sentences2)
sentences.extend(sentences3)
sentences.extend(sentences4)
sentences.extend(sentences5)


csvname1 = './data/IMDB/IMDB_label.csv'
csvname2 = './data/IMDB/IMDB_sentences.csv'
dataframe1 = pandas.DataFrame(label)
dataframe1.to_csv(csvname1, encoding='utf-8-sig')
dataframe2 = pandas.DataFrame(sentences)
dataframe2.to_csv(csvname2, encoding='utf-8-sig')
