import pandas as pd 
from tqdm import tqdm
#reading data
data=pd.read_csv(r"C:\Users\User\Desktop\Artificial intelligence\Machine learning\Deep learning\sentiment tweets\Data\training.1600000.processed.noemoticon.csv", encoding='latin-1')
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
dataaf.to_csv(r"C:\Users\User\Desktop\Artificial intelligence\Machine learning\Deep learning\sentiment tweets\Data\Clean_Data\Clean_data1.csv",index=False)


dataaf=pd.read_csv(r"C:\Users\User\Desktop\Artificial intelligence\Machine learning\Deep learning\sentiment tweets\Data\Clean_Data\Clean_data1.csv")



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

dataaf.to_csv(r"C:\Users\User\Desktop\Artificial intelligence\Machine learning\Deep learning\sentiment tweets\Data\Clean_Data\Clean_data2.csv",index=False)

dataaf=pd.read_csv(r"C:\Users\User\Desktop\Artificial intelligence\Machine learning\Deep learning\sentiment tweets\Data\Clean_Data\Clean_data2.csv")



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


dataaf.to_csv(r"C:\Users\User\Desktop\Artificial intelligence\Machine learning\Deep learning\sentiment tweets\Data\Clean_Data\Clean_data2.csv",index=False)



