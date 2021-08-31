import json
import numpy as np
import os
import pickle
import joblib
import time
from azureml.core.model import Model
import nltk
nltk.download("stopwords")
nltk.download("wordnet")
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
lemmatizer=WordNetLemmatizer()

def preprocess_data(data):
    corpus=[]
    for i in data:
        mess=re.sub("[^a-zA-Z0-9]"," ",i)
        mess=mess.lower().split()
        mess=[lemmatizer.lemmatize(word) for word in mess if word not in stopwords.words("english")]
        mess=" ".join(mess)
        corpus.append(mess)
    return corpus    


def init():
    global count_vect,rf_model
    
    count_vect_path=Model.get_model_path('NLP_Count_Vectorizer')
    count_vect= joblib.load(count_vect_path)
    
    rf_model_path=Model.get_model_path('NLP_RF_Model')
    rf_model=joblib.load(rf_model_path)
    
def run(raw_data):
    try:
        data = json.loads(raw_data)['data']
        corpus=preprocess_data(data[0])
        count_test=count_vect.transform(corpus)
        prediction=rf_model.predict(count_test)
        # you can return any data type as long as it is JSON-serializable
        return json.dumps({"result": prediction.tolist()})
    except Exception as e:
        result = str(e)
        # return error message back to the client
        return json.dumps({"error": result})
        
            