import time
import transformers


def get_bert_embedding(content, model, tokenizer):
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


def get_plot():
    title = "What is the title of your story?"
    villain = "Who/what is the villain in your story?"
    goal = "What is the villain's goal?"

    return title, villain, goal


def get_characters():
    creature = "What kind of fantasy creature is your character? (e.g., elf, orc, dwarf, human, etc.)?"
    archetype = "What archetype is your character? (e.g., hunter, wizard, warrior, etc.)?"
    companions = "How many travel companions does your character have?"

    return creature, archetype, companions


def get_setting():
    kingdom = "What kingdom is your story set in?"
    landscape = "Please describe the landscape"
    magic = "Is magic available in this universe?"

    return kingdom, landscape, magic


def get_sentiment():
    sentiment = "What is the sentiment of this story (e.g., sad, happy, hopeful, scary, adventurous, etc.)?"

    return sentiment


def main():
    input_list = []
    bert_encoding_vector = []
    bert_encoding_list = []

    # Intro
    print('Thank you for playing our interactive story adventure.')
    time.sleep(1)
    print('Please answer 10 questions to initialize your story.')
    time.sleep(1)
    print()

    # Plot
    title = input(get_plot()[0] + '\n')
    villain = input(get_plot()[1] + '\n')
    goal = input(get_plot()[2] + '\n')
    input_list.extend([title, villain, goal])
    print()

    # Characters
    creature = input(get_characters()[0] + '\n')
    archetype = input(get_characters()[1] + '\n')
    companions = input(get_characters()[2] + '\n')
    input_list.extend([creature, archetype, companions])
    print()

    # Setting
    kingdom = input(get_setting()[0] + '\n')
    landscape = input(get_setting()[1] + '\n')
    magic = input(get_setting()[2] + '\n')
    input_list.extend([kingdom, landscape, magic])
    print()

    # Sentiment
    sentiment = input(get_sentiment() + '\n')
    input_list.append(sentiment)
    print()

    # Thanks
    time.sleep(1)
    print('Thank you. Generating story...')

    # Get BERT embeddings for user inputs
    model = transformers.BertModel.from_pretrained('bert-base-uncased', return_dict=True).cuda()
    tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
    for content in input_list:
        output = list(get_bert_embedding(content, model, tokenizer))
        bert_encoding_vector.extend(output)
        bert_encoding_list.append((content, output))


if __name__ == "__main__":
    main()