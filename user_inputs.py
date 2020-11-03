import time
import transformers


class StoryInitializer(object):
    def __init__(self):
        # Initialize BERT model
        model = transformers.BertModel.from_pretrained('bert-base-uncased', return_dict=True).cuda()
        tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')

        self.content_list = []
        self.bert_encoding_vector = []

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
        self.title_bert = list(self.bert_embedding(self.title, model, tokenizer))
        self.villain = input(self.plot()[1] + '\n')
        self.villain_bert = list(self.bert_embedding(self.villain, model, tokenizer))
        self.villain_goal = input(self.plot()[2] + '\n')
        self.villain_goal_bert = list(self.bert_embedding(self.villain_goal, model, tokenizer))
        self.content_list.extend([("title", self.title, self.title_bert),
                                  ("villain", self.villain, self.villain_bert),
                                  ("villain_goal", self.villain_goal, self.villain_goal_bert)])
        print()

        # Characters
        self.creature = input(self.characters()[0] + '\n')
        self.creature_bert = list(self.bert_embedding(self.creature, model, tokenizer))
        self.archetype = input(self.characters()[1] + '\n')
        self.archetype_bert = list(self.bert_embedding(self.archetype, model, tokenizer))
        self.companions = input(self.characters()[2] + '\n')
        self.companions_bert = list(self.bert_embedding(self.companions, model, tokenizer))
        self.content_list.extend([("your_race", self.creature, self.creature_bert),
                                  ("your_archtype", self.archetype, self.archetype_bert),
                                  ("your_number_of_companions", self.companions, self.companions_bert)])
        print()

        # Setting
        self.kingdom = input(self.setting()[0] + '\n')
        self.kingdom_bert = list(self.bert_embedding(self.kingdom, model, tokenizer))
        self.landscape = input(self.setting()[1] + '\n')
        self.landscape_bert = list(self.bert_embedding(self.landscape, model, tokenizer))
        self.magic = input(self.setting()[2] + '\n')
        self.magic_bert = list(self.bert_embedding(self.magic, model, tokenizer))
        self.content_list.extend([("kingdom", self.kingdom, self.kingdom_bert),
                                  ("kingdom_landscape", self.landscape, self.landscape_bert),
                                  ("kingdom_magic", self.magic, self.magic_bert)])
        print()

        # Sentiment
        self.sentiment = input(self.story_sentiment() + '\n')
        self.sentiment_bert = list(self.bert_embedding(self.sentiment, model, tokenizer))
        self.content_list.append(("sentiment", self.sentiment, self.sentiment_bert))
        print()

        # Thanks
        time.sleep(1)
        print('Thank you. Generating story...')

        # Get BERT embeddings for user inputs
        for content_name, content, content_bert in self.content_list:
            self.bert_encoding_vector.extend(content_bert)

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
    print(story_initializer.content_list)


if __name__ == "__main__":
    main()