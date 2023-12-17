import json
from nltk.stem import PorterStemmer
from nltk.stem.lancaster import LancasterStemmer

import tensorflow as tf
from tensorflow.keras import models, layers
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import random

training, inference = False, True
#New Path: "C:\\Users\\duete\\OneDrive\\Desktop\\ChatBotProject\\intents.json"
with open("intents.json") as f:
    loaded_data = json.load(f)


def tokenize(sentence):
    tokenized = sentence.split()
    for i in range(len(tokenized)):
        if "?" in tokenized[i] or "." in tokenized[i] or "'" in tokenized[i] or "," in tokenized[i] or "!" in tokenized[i]:
            without_punct = tokenized[i][:-1]
            tokenized[i] = without_punct
    return tokenized

words = []
document_x = []
document_y = []
tags_list = []
stemmer = LancasterStemmer() 
for intent in loaded_data["intents"]:
    for pattern in intent["patterns"]:  
        tokenized = tokenize(pattern)
        stemmed_words = [stemmer.stem(w.lower()) for w in tokenized] #this stems all the tokenized words
        # document_x.append(pattern)
        document_x.append(stemmed_words) ########### edit
        document_y.append(intent["tag"])
        for w in stemmed_words: 
            words.append(w) #this appends all stemmed words into [word]
        tags_list.append(intent["tag"])


listset_words = list(set(words))   
listset_tags = list(set(tags_list))

listset_words = sorted(listset_words)
listset_tags = sorted(listset_tags)

print("words: ")
print(listset_words)#This is correct
print("tags: ")
print(listset_tags)
print("x doc: ")
print(document_x)
print("y doc: ")
print(document_y)

# set_words = [i, like, subject, secret, how, you]
# (sentence, tag) = (i like you, emotion)
# back of words -> return [1, 1, 0, 0, 0, 1]
# make a temp list that has length of set_words
# check if set_words[index] is in sentence
# temp_list[index]=1

def bagofwords(pattern):
    bow = [0] * len(listset_words)
    words_ = [stemmer.stem(w) for w in pattern]
    for i, word in enumerate(listset_words):

        if word in words_:
            bow[i] = 1
    return bow

def tagbow(tag):
    tbow = [0] * len(listset_tags)
    location = listset_tags.index(tag)
    tbow[location] = 1
    # tbow[listset_tags.index(tag)]=1
    return tbow

X = []
Y = []
for i, patterns in enumerate(document_x):
    entire_bow = bagofwords(patterns)
    X.append(entire_bow)
for tag in document_y: 
    entire_tbow = tagbow(tag)
    Y.append(entire_tbow)

print("x: ")
print(X)
print("y: ")
print(Y)
# print(a)



input_size = len(X[0])#length of word count
# Create a `Sequential` model and add a Dense layer as the first layer.
# model = tf.keras.models.Sequential()
# model.add(tf.keras.layers.Dense(64, input_shape = (input_size,), activation = 'relu'))
# model.add(tf.keras.layers.Dense(32, activation = 'relu'))
# model.add(tf.keras.layers.Dense(len(listset_tags), activation = 'softmax'))
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(len(X[0]))),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(len(Y[0]), activation='softmax')#Y[0] is the tags count
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())


#Adam is supposedly better than SGD(Stochastic Gradient Descent) but idk what these are in the first place 
#binary_crossentropy is not a good loss function as that handles only 2 classfications and we need a multiclass classification function 
# Multi-Class Cross-Entropy Loss: Used for when there are more than two categories 
# Sparse Multiclass Cross-Entropy Loss: similar to cross-entropy but uses integers as class labels
# Kullback Leibler Divergence Loss: More used for probability distributions? Which isn't the case here
# From these 3 Loss Functions(All used for multiclass) it seems multi-class cross entropy is best.

#The "Metric" segment is used to specify which evalution metrics you want to use. In this case, it calculates the classification accuracy 
# checkpoint_path = "C:\\Users\\duete\\OneDrive\\Desktop\\ChatBotProject\\chatterbot-corpus-master\\cp-{epoch:04d}.h5"
# if training == True:
    # model.fit(X,Y, epochs = 100, verbose=1)
model.fit(np.array(X), np.array(Y), epochs=250, batch_size=8, verbose=1) #########edit
    # model.save_weights(checkpoint_path.format(epoch=0))
# else:


while True:
    new_suserinput = []
    ui_bow = [] #userinput bag of words
    user_input = input("Begin with your sentence: ")
    if user_input.lower() == "quit": 
        break
    print("You said: "+ user_input)
    tokenized_ui = tokenize(user_input)
    stemmed_input = [stemmer.stem(w) for w in tokenized_ui]
    for i in stemmed_input:
        new_suserinput.append(i)
    print(new_suserinput)

    def newbow(new_suserinput):
        newbow = [0] * len(listset_words)
        words_ = [stemmer.stem(w) for w in new_suserinput]
        for i, word in enumerate(listset_words):
            if word in words_:
                newbow[i] = 1
        return np.array(newbow)
    ui_bow = newbow(new_suserinput)
    twodim_ui_bow = np.reshape(ui_bow,(1, len(ui_bow)))#reshaping so that it can be used for prediction
    # model.load_weights("C:\\Users\\duete\\OneDrive\\Desktop\\ChatBotProject\\chatterbot-corpus-master\\cp-0000.h5")
    print(np.array(twodim_ui_bow))
    result = model.predict(np.array(twodim_ui_bow))
    max_index = (np.argmax(result))
    print(result)
    print(max_index)
    tag = listset_tags[max_index]
    print(tag)
    for tg in loaded_data['intents']:
        if tg['tag'] == tag:
            responses = tg['responses']
            print("ChatBot: " + random.choice(responses))



