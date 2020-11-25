from empath import Empath
import nltk
from collections import defaultdict
nltk.download('averaged_perceptron_tagger')

class InputParser:
    def __init__(self):
        self.history = []
        self.nouns = set()
        self.characters = {"hero":["brave"]}#defaultdict(list)
        self.topics = []

    def parseInput(self, input):
        self.history.append(input)

        # topic modeling and additional topic generation
        lexicon = Empath()
        topicVector = lexicon.analyze(input, normalize=False)

        topics = []
        for key in topicVector.keys():
            if topicVector[key] > 0:
                topics.append(key)
        self.topics = topics
        return topics


    # def buildRelations(self, nouns, updates, kb):
    #     for n in nouns:
    #         if not kb[n]:
    #             kb[n] = updates
    #         else:
    #             kb[n].append(updates)
    #     return kb

    def identifyNouns(self, input):
        text = input.split(" ")
        POS = nltk.pos_tag(text)

        nouns = []
        for word in POS:
            if word[1] == "NN":
                nouns.append(word[0])
        # for noun in nouns:
        #     self.searchNouns(noun)
        self.nouns = self.nouns | set(nouns)
        return nouns

    # def searchNouns(self, noun):
    #     for topic in self.nouns.keys():
    #         if noun in self.nouns[topic]["synonyms"]:
    #             # we have a matching relation
    #             # we can have the relationship update here
    #             continue
    #         else:
    #             self.nouns[noun] = {"synonyms":[noun]}
    #             # we need some way of generating synonyms here

    def attributeParser(self, user_input):
        nouns = self.identifyNouns(user_input)
        for noun in nouns:
            if noun in self.characters.keys() and len(self.characters[noun]) < 2:
                print("Could you tell us a little bit more about ", noun, "? Say the ", noun, "is ___.")
                descriptiveInput = input()
                text = descriptiveInput.split(" ")
                idx = text.index("is")
                adjs = []
                for i in range(1, len(text)-idx):
                    POS = nltk.pos_tag(text)
                    if POS[idx+i][1] == "JJ" or POS[idx+i][1] == "NN":
                        self.characters[noun].append(POS[idx+i][0])
                # if idx < (len(text)-1) and text[idx+1] == "is":
                #     POS = nltk.pos_tag(text)
                #     for i in range(0, len(text)-idx):
                #         if POS[idx+i][1] == "JJ":
                #             self.characters[noun].append(POS[idx+i][0])
        return self.characters



if __name__ == '__main__':
    print("Enter text to be parsed:")
    user_input = input()

    ip = InputParser()
    print("Topics: ", ip.parseInput(user_input))
    print("Nouns: ", ip.identifyNouns(user_input))
    print("Known relations: ", ip.attributeParser(user_input))
