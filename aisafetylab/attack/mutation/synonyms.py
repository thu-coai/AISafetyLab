from aisafetylab.attack.mutation import BaseMutation
from aisafetylab.attack.selection import RouletteWheelSelection
import random
import re
import nltk
from nltk.corpus import stopwords, wordnet
from loguru import logger


class Synonyms(BaseMutation):
    def __init__(self, word_dict={}, crossover_rate=0.5):
        super().__init__()
        self.word_dict = word_dict
        self.crossover_rate = crossover_rate

    def update(self, word_dict):
        self.word_dict = word_dict
        
    def join_words_with_punctuation(self, words):
        sentence = words[0]
        previous_word = words[0]
        flag = 1
        for word in words[1:]:
            if word in [",", ".", "!", "?", ":", ";", ")", "]", "}", '”']:
                sentence += word
            else:
                if previous_word in ["[", "(", "'", '"', '“']:
                    if previous_word in ["'", '"'] and flag == 1:
                        sentence += " " + word
                    else:
                        sentence += word
                else:
                    if word in ["'", '"'] and flag == 1:
                        flag = 1 - flag
                        sentence += " " + word
                    elif word in ["'", '"'] and flag == 0:
                        flag = 1 - flag
                        sentence += word
                    else:
                        if "'" in word and re.search('[a-zA-Z]', word):
                            sentence += word
                        else:
                            sentence += " " + word
            previous_word = word
        return sentence

    def get_synonyms(self, word):
        synonyms = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonyms.add(lemma.name())
        return list(synonyms)

    def replace_synonyms(self, sentence):
        stop_words = set(stopwords.words('english'))
        T = {"llama2", "llama-2", "meta", "vicuna", "lmsys", "guanaco", "theblokeai", "wizardlm", "mpt-chat",
         "mosaicml", "mpt-instruct", "falcon", "tii", "chatgpt", "modelkeeper", "prompt"}
        words = nltk.word_tokenize(sentence)
        word_dict = self.word_dict
        # logger.debug(f'current word dict: {word_dict}')
        min_value = min(word_dict.values())
        count = 0
        for i, word in enumerate(words):
            if random.random() < self.crossover_rate:
                if word.lower() not in stop_words and word.lower() not in T:
                    synonyms = self.get_synonyms(word.lower())
                    word_scores = {
                        syn: word_dict.get(syn, min_value) for syn in synonyms
                    }
                    best_synonym = RouletteWheelSelection.word_roulette_wheel_selection(word, word_scores)
                    if best_synonym:
                        words[i] = best_synonym
                        count += 1
                        if count >= 5:
                            break
            else:
                if word.lower() not in stop_words and word.lower() not in T:
                    synonyms = self.get_synonyms(word.lower())
                    word_scores = {
                        syn: word_dict.get(syn, 0) for syn in synonyms
                    }
                    best_synonym = RouletteWheelSelection.word_roulette_wheel_selection(
                        word, word_scores
                    )
                    if best_synonym:
                        words[i] = best_synonym
                        count += 1
                        if count >= 5:
                            break
        
        return self.join_words_with_punctuation(words)
