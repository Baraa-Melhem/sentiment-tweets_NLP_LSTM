import numpy as np
from keras.models import load_model
import pickle
from keras.preprocessing.sequence import pad_sequences 



# recall the model
model1 = load_model(r'C:\Users\User\Desktop\Artificial intelligence\Machine learning\Deep learning\sentiment tweets\Model\MODEL.h5')


file_path = r"C:\Users\User\Desktop\Artificial intelligence\Machine learning\Deep learning\sentiment tweets\Model/tokenizer1.pickle"
with open(file_path, 'rb') as handle:
    loaded_tokenizer = pickle.load(handle)
    

#starting prediction

def NLP_on_text(txt):
        
    
    
    import neattext as nt
    import nltk
    mytext=txt
    docx = nt.TextFrame(text=mytext)
    docx.text 
    docx.text=docx.normalize(level='deep')
    docx=docx.remove_emojis()
    docx=docx.fix_contractions()
    txt=docx
    
    
    
    nwtxt=[]
    
    import spacy
    
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
    
    from nltk.stem import WordNetLemmatizer
    from nltk.corpus import wordnet
    from tqdm import tqdm
    
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
