from argparse import ArgumentParser
import os
from src.tasks.infer import infer_from_trained, FewRel
import sys
from knowledge_center import KnowledgeCenter

parser = ArgumentParser()
parser.add_argument("--task", type=str, default='semeval', help='semeval, fewrel')
parser.add_argument("--train_data", type=str, default=os.getcwd() + '/src/data/SemEval2010_task8_training/TRAIN_FILE.TXT', \
                    help="training data .txt file path")
parser.add_argument("--test_data", type=str, default=os.getcwd() + '/src/data/SemEval2010_task8_testing_keys/TEST_FILE_FULL.TXT', \
                    help="test data .txt file path")
parser.add_argument("--use_pretrained_blanks", type=int, default=0, help="0: Don't use pre-trained blanks model, 1: use pre-trained blanks model")
parser.add_argument("--num_classes", type=int, default=19, help='number of relation classes')
parser.add_argument("--batch_size", type=int, default=32, help="Training batch size")
parser.add_argument("--gradient_acc_steps", type=int, default=2, help="No. of steps of gradient accumulation")
parser.add_argument("--max_norm", type=float, default=1.0, help="Clipped gradient norm")
parser.add_argument("--fp16", type=int, default=0, help="1: use mixed precision ; 0: use floating point 32") # mixed precision doesn't seem to train well
parser.add_argument("--num_epochs", type=int, default=11, help="No of epochs")
parser.add_argument("--lr", type=float, default=0.00007, help="learning rate")
parser.add_argument("--model_no", type=int, default=0, help='''Model ID: 0 - BERT\n
                                                                        1 - ALBERT\n
                                                                        2 - BioBERT''')
parser.add_argument("--model_size", type=str, default='bert-base-uncased', help="For BERT: 'bert-base-uncased', \
                                                                                            'bert-large-uncased',\
                                                                                For ALBERT: 'albert-base-v2',\
                                                                                            'albert-large-v2'\
                                                                                For BioBERT: 'bert-base-uncased' (biobert_v1.1_pubmed)")
parser.add_argument("--train", type=int, default=0, help="0: Don't train, 1: train")
parser.add_argument("--infer", type=int, default=1, help="0: Don't infer, 1: Infer")

args = parser.parse_args(["--infer", "1"])

print(args)

inferer = infer_from_trained(args, detect_entities=True)
kc = KnowledgeCenter()

def characterBuilder(name):
    print("\n")
    print("Let's learn a bit more about %s" % name)
    gender = input("What is this character's gender?:\n")
    print("Got it")
    charType = input("What kind of character is %s? Please enter 'Hero', 'Villian', 'Main Hero', or 'Main Villian':\n" % name)
    print("Entering this character into the Knowledge Graph...")
    kc.createCharacterVertex({"Name": name, "Gender":gender, "Type":charType}, {})
print("\n")
print("===================================================================")
print("Let's create your characters.")
while True:
    currCharName = input("What is this character's name?:\n")
    characterBuilder(currCharName)
    more = input("Would you like to add another character? Please enter 'y' or 'n': \n")
    if more.lower() == "n":
        break

print("===================================================================")
print("Now let's describe some of the relations between the characters.")
print("This can include things like:\n 'So and so is an ally of so and so\n So and so uses so and so")

while True:
    sent = input("\n\nType input sentence ('quit' or 'exit' to terminate):\n")
    if sent.lower() in ['quit', 'exit']:
        kc.outputGraph("out_loop.png")
        break
    try:
        firstpass = kc.extractRelationTuple(sent)
        if None in firstpass:
            e1, e2, pred = inferer.infer_sentence(sent, detect_entities=True)
        else:
            e1, e2, pred = firstpass
        
        print("Entity 1: %s" % e1)
        print("Entity 2: %s" % e2)
        print("Captured Relation: %s" % pred)
        goodorbad = input("Is this extraction correct? If not type 'n' else hit enter")
        if goodorbad.lower() == "n":
            continue
        kc.createCharacterVertex({"Name": e1}, {})
        kc.createCharacterVertex({"Name": e2}, {})
        kc.createRelation(e1, e2, pred)
        print(pred)
        print(e1)
        print(e2)
    except:
        print("Try rephrasing that")
        print(sys.exc_info())
        continue