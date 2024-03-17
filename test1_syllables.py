import sys
import speech_recognition as sr
import nltk
import json
import Levenshtein as lev
import sounddevice as sd
import soundfile as sf
from nltk.corpus import cmudict

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
    print(f"Recording stopped and saved to {filename}")

def syllabify(word):
    if word.lower() in cmu_dict:
        syllables_phonemes = cmu_dict[word.lower()]
        syllables = []
        for phoneme_set in syllables_phonemes:
            word_syllables = []
            syllable = []
            for phoneme in phoneme_set:
                syllable.append(phoneme)
                if phoneme[-1].isdigit():  # Check if it's a vowel phoneme indicating a new syllable
                    word_syllables.append(''.join(syllable))
                    syllable = []
            syllables.append(word_syllables)
        return syllables
    else:
        print(f"No syllables found for word '{word}'")
        return [[word]]  # Return the word itself if not found in the dictionary

def text_to_phonemes(text):
    words = nltk.word_tokenize(text.lower())
    phonemes_list = []
    for word in words:
        phonemes = syllabify(word)
        if phonemes:
            phonemes_list.append(phonemes[0])
            # print(phonemes[0])
    return phonemes_list

def speech_to_text(audio_file):
    with sr.AudioFile(audio_file) as source:
        audio = recognizer.record(source)  # Read the entire audio file
    try:
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        return "Speech Recognition could not understand the audio"
    except sr.RequestError as e:
        return f"Could not request results; {e}"

def calculate_similarity(user_phoneme, corpus_phoneme):
    distance = lev.distance(str(user_phoneme), str(corpus_phoneme))
    length = max(len(user_phoneme), len(corpus_phoneme))
    similarity_score = (length - distance) / length
    return similarity_score

if __name__ == "__main__":
    audio_file_path = "user_audio.wav"
    record_audio(duration=7)

    transcribed_text = speech_to_text(audio_file_path)
    print(f"Transcribed Text: {transcribed_text}")

    user_phonemes_list = text_to_phonemes(transcribed_text)
    print(user_phonemes_list)

    ground_truth_sentence = 'The concert was canceled due to bad weather'
    gt_phonemes_list = text_to_phonemes(ground_truth_sentence)
    print(gt_phonemes_list)

    threshold = 0.75

    for user_word_syllables, gt_word_syllables in zip(user_phonemes_list, gt_phonemes_list):
        for i, (user_syllables, gt_syllables) in enumerate(zip(user_word_syllables, gt_word_syllables), start=1):
            similarity_score = calculate_similarity(' '.join(user_syllables), ' '.join(gt_syllables))
            print(f"Syllable {i} ('{' '.join(user_syllables)}' vs. '{' '.join(gt_syllables)}'):")
            print(f"Similarity Score: {similarity_score}")
            print(f"Encoded Score: {1 if similarity_score >= threshold else 0}")
