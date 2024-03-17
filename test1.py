import sys
import speech_recognition as sr
import nltk
import json
import nltk
from nltk.corpus import cmudict
import json
import Levenshtein as lev
import sounddevice as sd
import soundfile as sf



# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('cmudict')


# Initialize speech recognizer and CMU Pronouncing Dictionary
recognizer = sr.Recognizer()
cmu_dict = cmudict.dict()

def record_audio(duration, filename='user_audio.wav', fs=44100):
    print("Recording...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=2)
    sd.wait()
    sf.write(filename, recording, fs)
    print("Recording stopped and saved to", filename)

# Function to retrieve phonetic representation of a word
def get_phonetic_representation(word):
    if word.lower() in cmu_dict:
        return cmu_dict[word.lower()]
    else:
        return None


# Initialize speech recognizer
recognizer = sr.Recognizer()

# Function to convert text to phonemes
def text_to_phonemes(text):
    words = nltk.word_tokenize(text.lower())
    phonemes_list = []
    for word in words:
        if word in cmu_dict:
            phonemes = cmu_dict[word]
            phonemes_list.append(phonemes)
        else:
            phonemes_list.append([word])  # For words not in CMU dict, keep the word itself
    return phonemes_list

# Function to transcribe audio to text
def speech_to_text(audio_file):
    with sr.AudioFile(audio_file) as source:
        audio = recognizer.record(source)  # Read the entire audio file
    try:
        # Use Google Web Speech API for speech recognition
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        return "Speech Recognition could not understand the audio"
    except sr.RequestError as e:
        return f"Could not request results; {e}"

# Function to calculate similarity score between user phonemes and corpus phonemes
def calculate_similarity(user_phonemes, corpus_phonemes):
    # Consider only the 0th indexed phoneme
    user_phoneme = user_phonemes[0]
    corpus_phoneme = corpus_phonemes[0]

    # Compute Levenshtein distance between phonemes
    distance = lev.distance(user_phoneme, corpus_phoneme)
    length = max(len(user_phoneme), len(corpus_phoneme))
    similarity_score = (length - distance) / length

    return similarity_score

def get_phonemes(gt_sentence):
    words = gt_sentence.strip().split()
    with open('phonetic_representations.json') as f:
        ground_truth_data = json.load(f)

    final = []

    for word in words:
        phonemes_list = ground_truth_data.get(word.lower())
        if phonemes_list:
            final.append(phonemes_list)
        else:
            print(f"No phonemes found for word '{word}'")
    return final

if __name__ == "__main__":

    # Get the audio file path from command-line argument
    audio_file_path = "user_audio.wav"
    record_audio(duration=5)

    # Convert speech to text
    transcribed_text = speech_to_text(audio_file_path)
    print(f"Transcribed Text: {transcribed_text}")

    # Convert text to phonemes
    user_phonemes_list = text_to_phonemes(transcribed_text)
    print(f"User Phonemes: {user_phonemes_list}")

    
    ground_truth_sentence = 'the book is on the table'
    gt_phonemes_list = get_phonemes(ground_truth_sentence)
    print(f"GT Phonemes List: {gt_phonemes_list}")

    threshold = 0.65

    for user_phonemes, gt_phonemes in zip(user_phonemes_list, gt_phonemes_list):
        similarity_score = calculate_similarity(user_phonemes, gt_phonemes)
        print(f"Similarity Score: {similarity_score}")
        print(f"Encoded Score: {1 if similarity_score>=threshold else 0}")