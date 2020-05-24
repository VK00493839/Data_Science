# -*- coding: utf-8 -*-
"""
Created on Sun May 24 15:09:17 2020

@author: VINAY KUMAR REDDY
"""

# Both BOwW and TFIDF schemantic information is not stored. TFIDF gives importance
# to uncommon words. There is definitely chance of overfitting.

# In Word2Vec each word is represented as a vector of 32 or more dimensions instead 
# of a single number.

# Steps to create word2vec model
#------------------------------
# Tokenize the given sentences
# create Histograms
# Take Most frequent words
# create a matrix with all unique words. It also represents the occurence of relation
# between the words


import nltk, re
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from nltk import sent_tokenize, word_tokenize

paragraph = """I’m blaming Ducky for this. She came up with an idea for a game thread, that involves writing scenes from the middle of book. 
            A scene can be pages long, however. It seemed that something shorter might be just the thing. In this exercise, the idea is to write a paragraph that would be a random passage from a story. An effective paragraph is one that has unity (it isn’t a hodgepodge of things), focus (everything in the paragraph stacks up to the whatever-it-is the paragraph is about), and coherence (the content follows smoothly). 
            For this exercise, the paragraph should be quick to read--say, not be more than 100 words long. A paragraph needn’t be several sentences long, but might be only a sentence or two, or a single line of dialogue. Or it could be a snippet of dialogue with narration: She made an attempt to straighten her tawny hair. Her voice quavered with emotion. 
            “You must be a very lonely man, Judge Seagrave.” 
            Then she turned a gaze on him that might have ignited a rain-sodden haystack. “And I’m a lonely woman.” 
            It might be merely descriptive: Lines of weeds criss-crossed the cracked parking lot of the Seashell Motor Courts. 
            The flaking paint on the buildings had chalked to a pastel pink on walls covered with graffiti. 
            Many of the windows had been smashed out. 
            Where the sign had been, atop rusting steel posts, only the metal outline of a seashell remained. It might have action but no dialogue: It was Ms. Fitzhugh. She was walking fast. 
            A strange expression crossed the faces of the students as they glanced toward the door and saw the principal go straight into the boys’ restroom. The footsteps stopped. 
            There was a deep, throaty sound difficult to describe. 
            Then came an eruption of shrill screaming and a rapid sound of heels. 
            Moments later, Ms. Fitzhugh emerged, her eyes wild. 
            Screaming, she skidded in the hall and headed toward the office. 
            It might be expository: Above ground was the medieval settlement of Skaar’s Outpost, originally a fort to guard the cave entrance. 
            Its inception as a town had been in the lodging and supply needs of explorers there to attempt the subterranean labyrinth when it had opened as a commercial venture. 
            With the caverns’ flooding and subsequent closure, however, Skaar’s Outpost had declined into an agricultural community miles from any trade routes. 
            These are merely examples. Have fun!"""

# preprocessing the data            
text = re.sub(r'\[[0-9]*\]', ' ', paragraph)
text = re.sub(r'\s+', ' ', text)
text = text.lower()
text = re.sub(r'\d', ' ', text)
text = re.sub(r'\s+', ' ', text)

# tokenizing the data into sentences
sentences = sent_tokenize(text)

sentences = [word_tokenize(sentence) for sentence in sentences]

for i in range(len(sentences)):
    sentences[i] = [word for word in sentences[i] if word not in stopwords.words('english')]
    
# training word2vec model
model = Word2Vec(sentences, min_count=1) # if the word is less than 1 we skip the word

words = model.wv.vocab

# Finding word vectors
vector = model.wv['town']

# most similar words
similar = model.wv.most_similar('town')


    