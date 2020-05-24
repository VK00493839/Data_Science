# -*- coding: utf-8 -*-
"""
Created on Sun May 24 13:32:46 2020

@author: VINAY KUMAR REDDY
"""


# BOW wont give the importance for the specific word so we go for TFIDF 
# to give the importance for specific word in a given sentence


import nltk, re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
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
            
sentences = sent_tokenize(paragraph)  
ps = PorterStemmer()
lemm = WordNetLemmatizer()

corpus = []
for i in range(len(sentences)):
    review = re.sub('[^a-zA-Z]', " ", sentences[i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if word not in stopwords.words('english')]
    corpus.append(' '.join(review))

from sklearn.feature_extraction.text import CountVectorizer
# converts the words into vector form if word present in sent then mark as 1
# marks the word as number form how many times word present in sentence
cv = CountVectorizer(max_features=100) # max_features helps us to select the most important features
X = cv.fit_transform(corpus).toarray()   
    
