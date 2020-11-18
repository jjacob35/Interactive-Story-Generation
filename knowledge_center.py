from enum import Enum
from igraph import *
import time
import nltk 
import spacy 
nlp = spacy.load("en_core_web_sm")

class ProppStoryStage(Enum):
    INTRO = 1
    ABSEN = 2
    INTER = 3
    VIOLA = 4
    RECON = 5
    VILLL = 6
    MEDIA = 7
    PUNIS = 8
    WEDDD = 9
    FIN = 10

    def advance(self):
        if self.value < len(ProppStoryStage) - 1:
            return ProppStoryStage(self.value + 1)
        else:
            return ProppStoryStage(10)

class ProppStory():

    def __init__(self):
        self.currStage = ProppStoryStage.INTRO
        self.fields = self.getFields(self.currStage)
        
    def getFields(self, ProppStage):
        if ProppStage is ProppStoryStage.INTRO:
            newFields = {"Place": None, "Time": None,}
        elif ProppStage is ProppStoryStage.ABSEN:
            newFields = {
                "Motivation": None, # Why does hero want to achieve goal
                "Goal":None # what is the goal
            }
        elif ProppStage is ProppStoryStage.INTER:
            newFields = {
                "Warning":None, #What is the advice
                "Danger":None, #What is the danger/risk
                "Giver":None #Who gave the advice
            }
            
        elif ProppStage is ProppStoryStage.VIOLA:
            newFields = {
                "HeroAction" : None, #What does the hero do to violate the advice
            }
            
        elif ProppStage is ProppStoryStage.RECON:
            newFields = {
                "KnowledgeAboutHeroAcquired": None, #What has the villian learned about the hero
            }
            
        elif ProppStage is ProppStoryStage.VILLL:
            newFields = {
                "villiany": None, # True if villian does something
                "lack": None, # True if hero is missing something
                "villianAction":None, # What did the villian do
                "objectLacked":None # What is the hero missing
            }
            
        elif ProppStage is ProppStoryStage.MEDIA:
            newFields = {
                "heroPreperation":None, # What is the hero doing to prepare, maybe compensate for lack
            }
            
        elif ProppStage is ProppStoryStage.PUNIS:
            newFields = {
                "villianPunishment":None, # How do the heroes punish the villian
            }
        elif ProppStage is ProppStoryStage.WEDDD:
            newFields = {
                "heroReward":None, # What does the hero get
            }
        return newFields

    def advanceStory(self):
        self.currStage = self.currStage.advance()
        self.fields = self.getFields(self.currStage)

    def missingFields(self):
        missFields = []
        for field, value in self.fields.items():
            if not value:
                missFields.append(field)
        return missFields

class KnowledgeCenter:
    def __init__(self):
        self.charGraph = Graph(directed=True)
        self.charGraph.vs["Name"] = [""] # Character Name
        self.charGraph.vs["Gender"] = [""] #Character Gender
        self.charGraph.vs["Type"] = [""] # Character Type, hero, villian, etc.
        self.charGraph.es["Relation"] = [""]
        self.char2vertex = {}
    
    def createCharacterVertex(self, attributes, relations):
        self.charGraph.add_vertices(1)
        currVert = self.charGraph.vs[-1]
        for key, val in attributes.items():
            if key == "Name":
                self.char2vertex[val] = currVert
            currVert[key] = val
        
        for to, relation in relations.items():
            relVertex = self.char2vertex.get(to, None)
            if relVertex:
                relIndex = relVertex.index
                self.charGraph.add_edges([(currVert.index, relIndex)])
                relId = self.charGraph.get_eid(currVert.index, relIndex)
                self.charGraph.es[relId]["Relation"] = relation
        print("Character Added!")

    def createRelation(self, charSrc="", charDest="", relation=""):
        charSrcV = self.char2vertex.get(charSrc, None)
        charDestV = self.char2vertex.get(charDest, None)
        if charSrcV and charDestV:
            srcIndex = charSrcV.index
            destIndex = charDestV.index
            self.charGraph.add_edges([(srcIndex, destIndex)])
            relId = self.charGraph.get_eid(srcIndex, destIndex)
            self.charGraph.es[relId]["Relation"] = relation
            print("Relation Added")
        else:
            print("Missing Character")
        



    def outputGraph(self):
        layout = self.charGraph.layout("kk")
        self.charGraph.vs["label"] = self.charGraph.vs["Name"]
        self.charGraph.es["label"] = self.charGraph.es["Relation"]
        color_dict = {"Hero": "green", "Villian": "red", "Main Hero":"green", "Main Villian":"red"}
        self.charGraph.vs["color"] = [color_dict[charType] for charType in self.charGraph.vs["Type"]]
        plot(self.charGraph, target="output.png", layout = layout)
        # self.charGraph.write_svg("output" + str(time.time()).split(".")[0] + ".svg", layout=layout)

def extractRelationTuple(text):
    relationTuple = tuple()
    doc = nlp(text)
    for tok in doc:
        if tok.dep_ == "ROOT":
            relSrc = next(tok.lefts).text
            foundObj = None
            relation = tok.text
            updatedRel = False
            while not foundObj:
                currElem = next(tok.rights, None)
                if currElem and currElem.dep_ == "attr" and not updatedRel:
                    relation = currElem.text
                    updatedRel = True
                if not currElem:
                    break
                elif currElem.dep_.endswith("obj"):
                    foundObj = currElem.text
                else:
                    tok = currElem
            relDest = foundObj
            relationTuple = (relSrc, relDest, relation)
            break
    return relationTuple

if __name__ == "__main__":
    prop = ProppStory()

    print(prop.currStage)
    prop.advanceStory()
    print(prop.currStage)
    print(prop.missingFields())

    kc = KnowledgeCenter()
    kc.createCharacterVertex({"Name": "LukeSkywalker", "Gender":"M", "Type":"Main Hero"}, {})
    kc.createCharacterVertex({"Name": "DarthVader", "Gender":"M", "Type":"Main Villian"}, {})
    kc.createCharacterVertex({"Name": "HanSolo", "Gender":"M", "Type":"Hero"}, {})
    kc.createCharacterVertex({"Name": "Chewbacca", "Gender":"M", "Type":"Hero"}, {})
    relationTuple = extractRelationTuple("DarthVader is the father of LukeSkywalker")
    kc.createRelation(*relationTuple)
    kc.createRelation(*extractRelationTuple("HanSolo is an ally of LukeSkywalker."))
    kc.createRelation(*extractRelationTuple("HanSolo is an ally of Chewbacca."))
    kc.createRelation(*extractRelationTuple("Chewbacca is an ally of HanSolo"))
    kc.createRelation(*extractRelationTuple("Chewbacca is an ally of LukeSkywalker"))
    kc.createRelation(*extractRelationTuple("LukeSkywalker is an ally of HanSolo"))
    kc.createRelation(*extractRelationTuple("LukeSkywalker is an ally of Chewbacca"))
    # kc.createCharacterVertex({"Name": "Darth Vader", "Gender":"M", "Type":"Main Villian"}, {"Luke Skywalker": "Son"})
    # kc.createCharacterVertex({"Name": "Han Solo", "Gender":"M", "Type":"Hero"}, {"Luke Skywalker": "Ally"})
    # kc.createCharacterVertex({"Name": "Princess Leia", "Gender":"F", "Type":"Hero"}, {"Luke Skywalker": "Ally"})
    # kc.createCharacterVertex({"Name": "The Emperor", "Gender":"M", "Type":"Villian"}, {"Darth Vader": "Ally"})
    # kc.createCharacterVertex({"Name": "Chewbacca", "Gender":"M", "Type":"Hero"}, {"Han Solo": "Ally"})

    print(kc.char2vertex)
    print(kc.charGraph)
    kc.outputGraph()
