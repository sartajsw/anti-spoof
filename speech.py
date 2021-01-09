#!/usr/bin/env python3
# Requires PyAudio and PySpeech.
 
import speech_recognition as sr
import numpy as np

corpus = ['red', 'blue', 'green', 'yellow', 'white', 'home', 'eat', 'black', 'food', 'what', 'when', 'shirt', 'bell', 'system', 'rain']
text = " ".join(list(np.random.choice(corpus, 3)))

# Record Audio
r = sr.Recognizer()
with sr.Microphone() as source:
    print("Say: " + text)
    audio = r.listen(source,phrase_time_limit=10)
 
# Speech recognition using Google Speech Recognition
try:
    # for testing purposes, we're just using the default API key
    # to use another API key, use `r.recognize_google(audio, key="GOOGLE_SPEECH_RECOGNITION_API_KEY")`
    # instead of `r.recognize_google(audio)`
    sp = r.recognize_google(audio)
    print("You said: " + sp)
    if (sp == text):
        print("Please Enter!")
    else:
        print("Try again!")

except sr.UnknownValueError:
    print("Google Speech Recognition could not understand audio")

except sr.RequestError as e:
    print("Could not request results from Google Speech Recognition service; {0}".format(e))