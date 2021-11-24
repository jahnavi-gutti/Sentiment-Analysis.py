import tensorflow as tf
import tensorflow.keras as keras
import pandas as pd
import numpy as np
imdb_reviews=pd.read_csv("/content/imdb_reviews.csv")
test_reviews=pd.read_csv("/content/test_reviews.csv")
word_index=pd.read_csv("/content/word_indexes.csv")
word_index=dict(zip(word_index.Words,word_index.Indexes))
word_index["<PAD"]=0
word_index["<START"]=1
word_index["<UNK>"]=2
word_index["<UNUSED>"]=3
def review_encoder(text):
    arr=[word_index[word] for word in text]
    return arr
train_data,train_labels=imdb_reviews['Reviews'],imdb_reviews['Sentiment']
test_data,test_labels=test_reviews['Reviews'],test_reviews['Sentiment']
train_data=train_data.apply(lambda review:review.split())
test_data=test_data.apply(lambda review:review.split())
train_data=train_data.apply(review_encoder)
test_data=test_data.apply(review_encoder)
def encode_sentiments(sentiment):
    if sentiment=="postive":
        return 1
    else:
        return 0
train_labels=train_labels.apply(encode_sentiments)
test_labels=test_labels.apply(encode_sentiments)
train_data=keras.preprocessing.sequence.pad_sequences(train_data,value=word_index["<PAD"],padding='post',maxlen=500)
test_data=keras.preprocessing.sequence.pad_sequences(test_data,value=word_index["<PAD"],padding='post',maxlen=500)
model=keras.Sequential([keras.layers.Embedding(1000,16,input_length=500),
                        keras.layers.GlobalAveragePooling1D(),
                        keras.layers.Dense(16,activation='relu'),
                        keras.layers.Dense(1,activation='sigmoid')])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history=model.fit(train_data,train_labels,epochs=30,batch_size=512,validation_data=(test_data,test_labels))
loss,accuracy=model.evalute(test_data,test_labels)
index=np.random.randint(1,1000)
user_reviews=test_reviews.loc[index]
print(user_review)
user_review=test_data[index]
user_review=np.array([user_review])
if (model.predict(user_reviews)>0.5).astype("int32"):
    print("postive sentiment")
else:
    print("negative sentiment")
