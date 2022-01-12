from django.http import HttpResponse 
from django.shortcuts import render 
import io,os
import random
import string # to process standard python strings
import warnings
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')
from django.conf import settings

import nltk
from nltk.stem import WordNetLemmatizer
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('popular', quiet=True) 



with open(os.path.join(settings.BASE_DIR,'chatbot/static/data.txt'),encoding='utf8', errors ='ignore') as fin:
    raw = fin.read()

#TOkenisation
sent_tokens = nltk.sent_tokenize(raw)# converts to list of sentences 
word_tokens = nltk.word_tokenize(raw)# converts to list of words

# Preprocessing
lemmer = WordNetLemmatizer()
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.translate(remove_punct_dict)))


# Keyword Matching
GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up","hey",)
GREETING_RESPONSES = ["Nmeste", 'Hi, how can I help You?']

def greeting(sentence):
    """If user's input is a greeting, return a greeting response"""
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)


# Generating response
def response(user_response):
    robo_response=''
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx=vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    sent_tokens.remove(user_response)
    if(req_tfidf==0):
        robo_response=robo_response+"Kindly share your contact details and our admission team will get back to you."
        return robo_response
    else:
        robo_response = sent_tokens[idx].split('-')
        robo_response = "".join(str(x) for x in robo_response[1:])
        return robo_response



def home(request):
    return render(request,'index.html')

def chat(request):
    flag=True
    # return HttpResponse({"ROBO: My name is Robo. I will answer your queries about Chatbots. If you want to exit, type Bye!"})
    user_response = request.POST['user_input']
    user_response=user_response.lower()

    if(user_response!='bye'):
        if(user_response=='thanks' or user_response=='thank you' ):
            flag=False
            return HttpResponse("You're welcome..")
        else:
            if(greeting(user_response)!=None):
                return HttpResponse(greeting(user_response))
            else:
                return HttpResponse(response(user_response))
                # sent_tokens.remove(user_response)
    else:
        flag=False
        return HttpResponse("Bye! take care..")    
    