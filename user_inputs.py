import time
import transformers


class StoryInitializer(object):
    def __init__(self):
        self.input_list = []
        self.bert_encoding_vector = []
        self.bert_encoding_list = []

        # Plot
        self.title, self.title_bert = '', ''
        self.villain, self.villain_bert = '', ''
        self.villain_goal, self.villain_goal_bert = '', ''

        # Characters
        self.creature, self.creature_bert = '', ''
        self.archetype, self.archetype_bert = '', ''
        self.companions, self.companions_bert = '', ''

        # Setting
        self.kingdom, self.kingdom_bert = '', ''
        self.landscape, self.landscape_bert = '', ''
        self.magic, self.magic_bert = '', ''

        # Sentiment
        self.sentiment, self.sentiment_bert = '', ''

        # Intro
        print('Thank you for playing our interactive story adventure.')
        time.sleep(1)
        print('Please answer 10 questions to initialize your story.')
        time.sleep(1)
        print()

        # Plot
        self.title = input(self.plot()[0] + '\n')
        self.villain = input(self.plot()[1] + '\n')
        self.goal = input(self.plot()[2] + '\n')
        self.input_list.extend([self.title,
                                             self.villain,
                                             self.goal])
        print()

        # Characters
        self.creature = input(self.characters()[0] + '\n')
        self.archetype = input(self.characters()[1] + '\n')
        self.companions = input(self.characters()[2] + '\n')
        self.input_list.extend([self.creature,
                                             self.archetype,
                                             self.companions])
        print()

        # Setting
        self.kingdom = input(self.setting()[0] + '\n')
        self.landscape = input(self.setting()[1] + '\n')
        self.magic = input(self.setting()[2] + '\n')
        self.input_list.extend([self.kingdom,
                                             self.landscape,
                                             self.magic])
        print()

        # Sentiment
        self.sentiment = input(self.story_sentiment() + '\n')
        self.input_list.append(self.sentiment)
        print()

        # Thanks
        time.sleep(1)
        print('Thank you. Generating story...')

        # Get BERT embeddings for user inputs
        model = transformers.BertModel.from_pretrained('bert-base-uncased', return_dict=True).cuda()
        tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
        for content in self.input_list:
            output = list(self.bert_embedding(content, model, tokenizer))
            self.bert_encoding_vector.extend(output)
            self.bert_encoding_list.append((content, output))


    def bert_embedding(self, content, model, tokenizer):
        encoding = tokenizer.encode_plus(
            content,
            add_special_tokens=True,
            padding='max_length',
            return_attention_mask=True,
            return_token_type_ids=False,
            return_tensors='pt')

        output = model(encoding['input_ids'].cuda(), encoding['attention_mask'].cuda())
        vector = output[0][0][0].detach().cpu().numpy()

        return vector

    def plot(self):
        title = "What is the title of your story?"
        villain = "Who/what is the villain in your story?"
        goal = "What is the villain's goal?"

        return title, villain, goal

    def characters(self):
        creature = "What kind of fantasy creature is your character? (e.g., elf, orc, dwarf, human, etc.)?"
        archetype = "What archetype is your character? (e.g., hunter, wizard, warrior, etc.)?"
        companions = "How many travel companions does your character have?"

        return creature, archetype, companions

    def setting(self):
        kingdom = "What kingdom is your story set in?"
        landscape = "Please describe the landscape"
        magic = "Is magic available in this universe?"

        return kingdom, landscape, magic

    def story_sentiment(self):
        sentiment = "What is the sentiment of this story (e.g., sad, happy, hopeful, scary, adventurous, etc.)?"

        return sentiment


def main():
    story_initializer = StoryInitializer()


if __name__ == "__main__":
    main()