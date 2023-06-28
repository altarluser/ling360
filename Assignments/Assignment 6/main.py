import streamlit as st
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from scipy.spatial.distance import cosine
from sklearn.metrics import pairwise_distances

def getSongRecommendation(text, topK=10):
  """Function returns a 
  
  :args:
      text (str): text of explaining current mood
      topK (int): number of song titles to be returned
      
  :returns:
      top_tracks (list): list containing `topK` number of song titles
  """
  
  bow = st.session_state["bow"]
  model = st.session_state["model"]
  
  X_text = st.session_state["X_text"]
  X_mood = st.session_state["X_mood"]
  
  ## YOUR CODE BELOW AT QUESTION 1.5
  x_text = bow.transform([text])
  x_mood = model.predict_proba(x_text)

  sim_mood = calculateSimilarity(x_mood, X_mood)
  sim_text = calculateSimilarity(x_text, X_text)
  
  mean_scores = []
  for mood, text in list(zip(sim_mood, sim_text)):
      mean_score = (mood + text) / 2 
      mean_scores.append(mean_score)

  scores = dict(zip(song_titles, mean_scores))
  sorted_scores = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))

  top_tracks = [(key, sorted_scores[key]) for key in list(sorted_scores.keys())[:topK]]
  
  ## YOUR CODE ABOVE AT QUESTION 1.5
  
  return top_tracks


def calculateSimilarity(document_vector, all_vectors):
    """The function that checks similarity between input and all instance of features of corpus
    
    """
    return (1 - pairwise_distances(all_vectors,document_vector, metric="cosine")).T[0]

if "X_mood" in st.session_state.keys():
    
  ## COPY YOUR BELOW HERE (Q1.1 TO Q1.4)

  train_emotion = 'train_emotion.csv'
  test_emotion = 'test_emotion.csv'

  columns = ['sentence', 'label']
  train_emotion_data = pd.read_csv(train_emotion, sep='\t', names=columns, encoding="utf-8")
  test_emotion_data = pd.read_csv(test_emotion, sep='\t', names=columns, encoding="utf-8")

  train_sentences = train_emotion_data['sentence']
  test_sentences = test_emotion_data['sentence']

  train_labels = train_emotion_data['label']
  test_labels = test_emotion_data['label']

  label_to_int = {'sadness':0,'joy':1,'love':2,'anger':3,'fear':4,'surprise':5}
  int_to_label = {0:'sadness',1:'joy',2:'love',3:'anger',4:'fear',5:'surprise'}

  bow = CountVectorizer() 

  X_train = bow.fit_transform(train_sentences)
  model = MultinomialNB()
  model.fit(X_train, train_labels)

  song_dataset = 'song_dataset.csv'
  song_dataset_data = pd.read_csv(song_dataset, sep='\t', names=['song_name', 'lyrics'], encoding="utf-8")

  song_titles = song_dataset_data['song_name']
  song_lyrics = song_dataset_data['lyrics']

  X_text = bow.transform(song_lyrics)
  X_mood = model.predict_proba(X_text)

  ## COPY YOUR ABOVE HERE (Q1.1 TO Q1.4)    

  st.session_state["bow"] = bow
  st.session_state["model"] = model
  st.session_state["X_text"] = X_text
  st.session_state["X_mood"] = X_mood
    
######################
# MAIN USER INTERFACE
######################

# Define input_text component
input_text = st.text_input("How are you?:", "")

# Define button here
button = st.button("Calculate")

if button:
  model = st.session_state["model"]
  X_mood = st.session_state["X_mood"]

  input_vector = bow.transform([input_text])
  probabilities = model.predict_proba(input_vector)[0]

  max_prob = -1
  max_prob_index = -1
  for i in range(len(probabilities)):
    if probabilities[i] > max_prob:
      max_prob = probabilities[i]
      max_prob_index = i
  
  model_prediction = int_to_label[max_prob_index]

# Get model prediction and display "You feel `model_prediction`" via `st.write()`
  st.write("You feel:", model_prediction)
  
  # Using for loop, display "Your songs..\n 1. song_title, 2. song_title....." via `st.write()` as in diagram in question.
  recommendations = getSongRecommendation(input_text)
  for i, (title, score) in enumerate(recommendations, start=1):
    st.write(f"{i}. {title}")
    