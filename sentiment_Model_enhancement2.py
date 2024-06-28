import pandas as pd 
from tqdm import tqdm
import numpy as np
#reading data
data=pd.read_csv(r"C:\Users\User0\Desktop\Artificial intelligence\Machine learning\Deep learning\sentiment tweets\Data\training.1600000.processed.noemoticon.csv", encoding='latin-1')
data.columns
# simple data cleaning before starting in NLP
data.describe(include="all")

data.drop("1467810369",axis=1,inplace=True)  #If you want to drop any column
data.drop("Mon Apr 06 22:19:45 PDT 2009",axis=1,inplace=True)  #If you want to drop any column
data.drop("NO_QUERY",axis=1,inplace=True)  #If you want to drop any column
data.drop("_TheSpecialOne_",axis=1,inplace=True)  #If you want to drop any column

data.rename(columns={'0': 'label'}, inplace=True) #rename columns
data.rename(columns={"@switchfoot http://twitpic.com/2y1zl - Awww, that's a bummer.  You shoulda got David Carr of Third Day to do it. ;D": 'content'}, inplace=True)

data.head()

data.dropna(axis=0,inplace=True)
data.isna().sum()






### NLP ###

# text cleaning 


# text cleaning libs
import neattext as nt
import nltk
#nltk.download()  # in case you did not download it here is the steps :remove the "#" before "nltk.download()" ,then choose : 1>>d 2>>l 3>>all

total_rows = 1600000 # this is the number of raws make sure to put the right number
cind=1 # the index of the column that you want to apply NLP on
with tqdm(total=total_rows) as pbar:
    for i in range(len(data["content"])):
        mytext=data.iloc[i,cind] 
        docx = nt.TextFrame(text=mytext)
        docx.text 
        docx.text=docx.normalize(level='deep')
        docx=docx.remove_emojis()
        docx=docx.fix_contractions()
        data.iloc[i,cind]=docx
        pbar.update(1)

print("Data cleaning, completed.")




#!pip install neattext

# POS tagging
#it could be :
# [noun
# verb
# adjective
# adverb
# pronoun
# determiner
# conjunction
# preposition
# interjection
# common noun
# proper noun
# mass noun
# count noun

cind=1 # the index of the column that you want to apply NLP on 
dataaf=data

nwtxt=[]

import spacy

nlp = spacy.load('en_core_web_sm')
newcleantext = []

with tqdm(total=total_rows) as pbar:
    for i in range(len(dataaf["content"])):
        doc1 = nlp(dataaf.iloc[i, cind])
        postex = []
        
        for token in doc1:
            wordtext = token.text
            poswrd = spacy.explain(token.pos_)

            # Map POS to single characters
            if poswrd == "verb":
                poswrd = "v"
            elif poswrd == "noun":
                poswrd = "n"
            elif poswrd == "adjective":
                poswrd = "a"
            elif poswrd == "adverb":
                poswrd = "r"
            elif poswrd == "pronoun":
                poswrd = "n"
            elif poswrd == "determiner":
                poswrd = "dt"
            elif poswrd == "conjunction":
                poswrd = "cc"
            elif poswrd == "preposition":
                poswrd = "prep"
            elif poswrd == "interjection":
                poswrd = "intj"
            elif poswrd == "common noun":
                poswrd = "n"
            elif poswrd == "proper noun":
                poswrd = "n"
            elif poswrd == "mass noun":
                poswrd = "n"
            elif poswrd == "count noun":
                poswrd = "n"
            else:
                poswrd = "n"

            postex.append(f"({wordtext})({poswrd})")

        newcleantext.append(",".join(postex))
        pbar.update(1)

lemmtext = {"cleantext2": newcleantext}
newtextline = pd.DataFrame(lemmtext)
print("Pos tagging, completed..")

dataaf=pd.concat([dataaf,newtextline],axis=1)

####
dataaf.to_csv(r"C:\Users\User0\Desktop\Artificial intelligence\Machine learning\Deep learning\sentiment tweets\Data\New clean data\Clean_data1.csv",index=False)


dataaf=pd.read_csv(r"C:\Users\User0\Desktop\Artificial intelligence\Machine learning\Deep learning\sentiment tweets\Data\New clean data\Clean_data1.csv")



#lemmatization
# pos parameter
# "n" for nouns
# "v" for verbs
# "a" for adjectives
# "r" for adverbs
# "s" for satellite adjectives
# Determiner	dt
# Conjunction	cc
# Preposition	prep
# Interjection	intj
# Noun	n
# pos=wordnet.NOUN

dataaf.isna().sum()


from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from tqdm import tqdm

newcleantext = []
lemmatizer = WordNetLemmatizer()
dataaf.dropna(axis=0, inplace=True)

# after any 'dataaf.dropna(axis=0, inplace=True)' you need to make sure to enter the right total raws

cind=2 # the index of the column that you want to apply NLP on 

with tqdm(total=total_rows) as pbar:
    for i in range(len(dataaf["cleantext2"])):
        textsve = ""
        text_in_data = dataaf.iloc[i, cind]
        tokens = [pair.strip("()").split("),") for pair in text_in_data.split("),(")]
        for word_pos in tokens:
            if len(word_pos) == 1:
                word, pos = word_pos[0], 'n'  #  'n' as a default part of speech if there's only one value.
            else:
                word, pos = word_pos
                    
            if pos == "dt" or pos == "cc" or pos == "prep" or pos == "intj":
                pos = wordnet.NOUN
            
            textsve = lemmatizer.lemmatize(word, pos=pos) + " " + textsve
        

        newcleantext.append(textsve)
        pbar.update(1)

lemmtext = {"cleantext": newcleantext}
lemmtext = pd.DataFrame(lemmtext)
print("lemmatization, completed..")

dataaf=dataaf.drop("cleantext2",axis=1)
dataaf=pd.concat([dataaf,lemmtext],axis=1)

dataaf.to_csv(r"C:\Users\User0\Desktop\Artificial intelligence\Machine learning\Deep learning\sentiment tweets\Data\New clean data\Clean_data2.csv",index=False)

dataaf=pd.read_csv(r"C:\Users\User0\Desktop\Artificial intelligence\Machine learning\Deep learning\sentiment tweets\Data\New clean data\Clean_data2.csv")



###
with tqdm(total=total_rows) as pbar:
    for i in range(len(dataaf["cleantext"])):
        text=dataaf.iloc[i,2]
        text_without_parentheses = text.replace("(v", "").replace(")", "").replace("(r","").replace("(n","").replace("(a","").replace("(dt","").replace("(cc","").replace("(prep","").replace("(intj","")
        dataaf.iloc[i,2]=text_without_parentheses
        pbar.update(1)
        
        

# text cleaning2 

cind=2 # the index of the column that you want to apply NLP on
with tqdm(total=total_rows) as pbar:
    for i in range(len(dataaf["cleantext"])):
        mytext=dataaf.iloc[i,cind] 
        docx = nt.TextFrame(text=mytext)
        docx.text 
        docx=docx.remove_stopwords()
        docx=docx.fix_contractions()
        dataaf.iloc[i,cind]=docx
        pbar.update(1)


print("Data cleaning completed.")        
   


     

# Reverse the order of words

cind=2 # the index of the column that you want to apply NLP on
with tqdm(total=total_rows) as pbar:
    for i in range(len(dataaf["cleantext"])):
        text=dataaf.iloc[i,cind] 
        words = text.split()
        reversed_text = ' '.join(words[::-1])
        dataaf.iloc[i,cind]=reversed_text
        pbar.update(1)


dataaf.to_csv(r"C:\Users\User0\Desktop\Artificial intelligence\Machine learning\Deep learning\sentiment tweets\Data\New clean data\Clean_data2.csv",index=False)
















#### AI Model

dataaf.isna().sum()
dataaf.dropna(axis=0, inplace=True)


# Label encoder to "label" 
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
dataaf["label"] = label_encoder.fit_transform(dataaf["label"])

class_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print("Class Label Mapping:", class_mapping)

dataaf.drop("content",axis=1,inplace=True)

dataaf["label"].value_counts()
dataaf.isna().sum()


from sklearn.utils import shuffle
dataaf = shuffle(dataaf)

        
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.models import Sequential
from keras.regularizers import l2
from keras.utils import to_categorical


x = dataaf["cleantext"]
y = dataaf["label"].values # Convert to a NumPy array
tokenizer = Tokenizer()
tokenizer.fit_on_texts(x)
sequences = tokenizer.texts_to_sequences(x)

word_index = tokenizer.word_index
vocab_size = len(word_index) + 1
max_sequence_length = 15 #to control the length of the sequences
padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)

from keras.layers import Conv1D, MaxPooling1D
from keras.layers import BatchNormalization
from keras.layers import Activation



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, y, test_size=0.2, random_state=42)


num_classes = len(np.unique(y))# Number of classes

embedding_dim = 40
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_sequence_length))

model.add(LSTM(units=60, return_sequences=True, kernel_regularizer=l2(0.02)))
model.add(Dropout(0.3))
model.add(LSTM(units=30, kernel_regularizer=l2(0.02)))

model.add(Dense(units=num_classes, activation='sigmoid', kernel_regularizer=l2(0.05)))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

y_train_enc = to_categorical(y_train, num_classes=num_classes)
y_test_enc = to_categorical(y_test, num_classes=num_classes)

# Convert data types
X_train = np.asarray(X_train).astype(np.float32)
y_train_enc = np.asarray(y_train_enc).astype(np.int32)
X_test = np.asarray(X_test).astype(np.float32)
y_test_enc = np.asarray(y_test_enc).astype(np.int32)


model.fit(X_train, y_train_enc, batch_size=1000, epochs=7)
accuracy = model.evaluate(X_test, y_test_enc, verbose=0)
print("Accuracy:", accuracy)


# saving the model
model.save(r"C:\Users\User0\Desktop\Artificial intelligence\Machine learning\Deep learning\sentiment tweets\Model-2\Model\MODEL.h5")

import pickle
file_path = r"C:\Users\User0\Desktop\Artificial intelligence\Machine learning\Deep learning\sentiment tweets\Model-2/tokenizer1.pickle"
with open(file_path, 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    
    
    
    


# Trying model

from keras.models import load_model
import pickle
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

import spacy

nlp = spacy.load('en_core_web_sm')
import neattext as nt
import nltk
import pandas as pd 
from tqdm import tqdm
import numpy as np


# recall the model
model1 = load_model(r"C:\Users\User0\Desktop\Artificial intelligence\Machine learning\Deep learning\sentiment tweets\Model-2\Model\MODEL.h5")


file_path = r"C:\Users\User0\Desktop\Artificial intelligence\Machine learning\Deep learning\sentiment tweets\Model-2/tokenizer1.pickle"
with open(file_path, 'rb') as handle:
    loaded_tokenizer = pickle.load(handle)
    

#starting prediction

def NLP_on_text(txt):
        
    
    

    mytext=txt
    docx = nt.TextFrame(text=mytext)
    docx.text 
    docx.text=docx.normalize(level='deep')
    docx=docx.remove_emojis()
    docx=docx.fix_contractions()
    txt=docx
    
    
    
    nwtxt=[]
    
    
    nlp = spacy.load('en_core_web_sm')
    newcleantext = []
    
    
    doc1 = nlp(txt)
    postex = []
    
    for token in doc1:
        wordtext = token.text
        poswrd = spacy.explain(token.pos_)
    
        # Map POS to single characters
        if poswrd == "verb":
            poswrd = "v"
        elif poswrd == "noun":
            poswrd = "n"
        elif poswrd == "adjective":
            poswrd = "a"
        elif poswrd == "adverb":
            poswrd = "r"
        elif poswrd == "pronoun":
            poswrd = "n"
        elif poswrd == "determiner":
            poswrd = "dt"
        elif poswrd == "conjunction":
            poswrd = "cc"
        elif poswrd == "preposition":
            poswrd = "prep"
        elif poswrd == "interjection":
            poswrd = "intj"
        elif poswrd == "common noun":
            poswrd = "n"
        elif poswrd == "proper noun":
            poswrd = "n"
        elif poswrd == "mass noun":
            poswrd = "n"
        elif poswrd == "count noun":
            poswrd = "n"
        else:
            poswrd = "n"
    
        postex.append(f"({wordtext})({poswrd})")
    
    newcleantext.append(",".join(postex))
    

    
    lemmatizer = WordNetLemmatizer()
    
    textsve = ""
    text_in_data = txt
    tokens = [pair.strip("()").split("),") for pair in text_in_data.split("),(")]
    for word_pos in tokens:
        if len(word_pos) == 1:
            word, pos = word_pos[0], 'n'  #  'n' as a default part of speech if there's only one value.
        else:
            word, pos = word_pos
                
        if pos == "dt" or pos == "cc" or pos == "prep" or pos == "intj":
            pos = wordnet.NOUN
        
        textsve = lemmatizer.lemmatize(word, pos=pos) + " " + textsve
    
    newcleantext = []
    newcleantext.append(textsve)
    
    
    
    
    
    ###
    
    text=newcleantext[0]
    text_without_parentheses = text.replace("(v", "").replace(")", "").replace("(r","").replace("(n","").replace("(a","").replace("(dt","").replace("(cc","").replace("(prep","").replace("(intj","")
    newcleantext=text_without_parentheses
            
            
    
    # text cleaning2 
    
    mytext=newcleantext
    docx = nt.TextFrame(text=mytext)
    docx.text 
    docx=docx.remove_stopwords()
    docx=docx.fix_contractions()
    newcleantext=docx
       

    # Reverse the order of words
    
    text=newcleantext
    words = text.split()
    reversed_text = ' '.join(words[::-1])
    newcleantext=reversed_text
    
    return newcleantext


txt=input("text : ")
print("\n")

newcleantext=NLP_on_text(txt)
print("clean text:",newcleantext,"\n")
sequences = loaded_tokenizer.texts_to_sequences([newcleantext])
max_sequence_length = 15  
padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)

predictions = model1.predict(padded_sequences)


predictedind1= np.argmax(predictions[0])# index of the highest probability

if predictedind1==0:
    predictedind1="negative"
elif predictedind1==1:
    predictedind1="positive"


print("\n text sentiment :", predictedind1)    