import time
import transformers
import pprint


class StoryInitializer(object):
    def __init__(self):
        # Initialize BERT model
        model = transformers.BertModel.from_pretrained('bert-base-uncased', return_dict=True)
        tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')

        self.content_dictionary = {}

        # Introduction
        self.intro = {}
        self.title, self.title_bert = '', ''
        self.kingdom, self.kingdom_bert = '', ''
        self.landscape, self.landscape_bert = '', ''
        self.magic, self.magic_bert = '', ''
        self.sentiment, self.sentiment_bert = '', ''

        # Absentation
        self.absent = {}
        self.goal, self.goal_bert = '', ''
        self.motivation, self.motivation_bert = '', ''

        # Interdiction
        self.interdict = {}
        self.danger, self.danger_bert = '', ''
        self.warning, self.warning_bert = '', ''
        self.giver, self.giver_bert = '', ''

        # Violation
        self.violation = {}
        self.hero_action, self.hero_action_bert = '', ''

        # Recon
        self.recon = {}
        self.knowledge_about_hero_acquired, self.knowledge_about_hero_acquired = '', ''

        # Villainy or lack of villainy
        self.villorlack = {}
        self.villainy, self.villainy_bert = '', ''
        self.lack, self.lack_bert = '', ''
        self.villain, self.villain_bert = '', ''
        self.villain_goal, self.villain_goal_bert = '', ''
        self.villain_action, self.villain_action_bert = '', ''
        self.object_lacked, self.object_lacked_bert = '', ''

        # Mediation
        self.mediate = {}
        self.hero_preparation, self.hero_preparation_bert = '', ''

        # Punishment
        self.punish = {}
        self.villain_punishment, self.villain_punishment_bert = '', ''

        # Wedding (or other positive event for the heroes)
        self.wed = {}
        self.hero_reward, self.hero_reward_bert = '', ''

        # Character Info
        self.charinfo = {}
        self.creature, self.creature_bert = '', ''
        self.archetype, self.archetype_bert = '', ''
        self.companions, self.companions_bert = '', ''


        # Player Intro
        print('Thank you for playing our interactive story adventure.')
        time.sleep(1)
        print('Please answer 24 questions to initialize your story. Press enter to skip a question.')
        time.sleep(1)
        print()


        # Introduction
        print('First, we will ask you some questions about your story setting.')
        time.sleep(2)
        self.title = input(self.introduction()[0] + '\n')
        self.title_bert = list(self.bert_embedding(self.title, model, tokenizer))
        self.intro['Title'] = self.title  # , self.title_bert

        self.kingdom = input(self.introduction()[1] + '\n')
        self.kingdom_bert = list(self.bert_embedding(self.kingdom, model, tokenizer))
        self.intro['Kingdom'] = self.kingdom  # , self.kingdom_bert

        self.landscape = input(self.introduction()[2] + '\n')
        self.landscape_bert = list(self.bert_embedding(self.landscape, model, tokenizer))
        self.intro['Landscape'] = self.landscape  # , self.landscape_bert

        self.magic = input(self.introduction()[3] + '\n')
        self.magic_bert = list(self.bert_embedding(self.magic, model, tokenizer))
        self.intro['Magic'] = self.magic  # , self.magic_bert

        self.sentiment = input(self.introduction()[4] + '\n')
        self.sentiment_bert = list(self.bert_embedding(self.sentiment, model, tokenizer))
        self.intro['Sentiment'] = self.sentiment  # , self.sentiment_bert

        self.content_dictionary['Intro'] = self.intro

        print()


        # Character Info
        print('Next, we will ask you some questions about your character.')
        time.sleep(2)
        self.creature = input(self.characterinfo()[0] + '\n')
        self.creature_bert = list(self.bert_embedding(self.creature, model, tokenizer))
        self.charinfo['Creature'] = self.creature  # , self.creature_bert

        self.archetype = input(self.characterinfo()[1] + '\n')
        self.archetype_bert = list(self.bert_embedding(self.archetype, model, tokenizer))
        self.charinfo['Archetype'] = self.archetype  # , self.archetype_bert

        self.companions = input(self.characterinfo()[2] + '\n')
        self.companions_bert = list(self.bert_embedding(self.companions, model, tokenizer))
        self.charinfo['Companions'] = self.companions  # , self.companions_bert

        self.content_dictionary['CharacterInfo'] = self.charinfo

        print()


        # Absentation
        print('Next, we want to understand more about your character\'s motivations in this story.')
        time.sleep(2)
        self.goal = input(self.absentation()[0] + '\n')
        self.goal_bert = list(self.bert_embedding(self.goal, model, tokenizer))
        self.absent['Goal'] = self.goal  # , self.goal_bert

        self.motivation = input(self.absentation()[1] + '\n')
        self.motivation_bert = list(self.bert_embedding(self.motivation, model, tokenizer))
        self.absent['Motivation'] = self.motivation  # , self.motivation_bert

        self.content_dictionary['Absentation'] = self.absent

        print()

        # Interdiction
        print('Next are some questions about the danger your character will face.')
        time.sleep(2)
        self.danger = input(self.interdiction()[0] + '\n')
        self.danger_bert = list(self.bert_embedding(self.danger, model, tokenizer))
        self.interdict['Danger'] = self.danger  # , self.danger_bert

        self.warning = input(self.interdiction()[1] + '\n')
        self.warning_bert = list(self.bert_embedding(self.warning, model, tokenizer))
        self.interdict['Warning'] = self.warning  # , self.warning_bert

        self.giver = input(self.interdiction()[2] + '\n')
        self.giver_bert = list(self.bert_embedding(self.giver, model, tokenizer))
        self.interdict['Giver'] = self.giver  # , self.giver_bert

        self.content_dictionary['Interdiction'] = self.interdict

        print()

        # Violation
        print('But, your character may not listen to the advice given...')
        time.sleep(2)
        self.hero_action = input(self.violate_advice() + '\n')
        self.hero_action_bert = list(self.bert_embedding(self.hero_action, model, tokenizer))
        self.violation['HeroAction'] = self.hero_action  # , self.hero_action_bert

        self.content_dictionary['Violation'] = self.violation

        print()

        # Recon
        print('The villain has also been investigating your character.')
        time.sleep(2)
        self.knowledge_about_hero_acquired = input(self.reconaissance() + '\n')
        self.knowledge_about_hero_acquired_bert = list(self.bert_embedding(self.knowledge_about_hero_acquired, model, tokenizer))
        self.recon['HeroAction'] = self.hero_action  # , self.hero_action_bert

        self.content_dictionary['Recon'] = self.recon

        print()

        # Villainy or lack of villainy
        print('Please answer some questions about the villain in this story.')
        time.sleep(2)
        self.villainy = input(self.villainorlack()[0] + '\n')
        self.villainy_bert = list(self.bert_embedding(self.villainy, model, tokenizer))
        self.villorlack['Villainy'] = self.villainy  # , self.villainy_bert

        self.lack = input(self.villainorlack()[1] + '\n')
        self.lack_bert = list(self.bert_embedding(self.lack, model, tokenizer))
        self.villorlack['Lack'] = self.lack  # , self.lack_bert

        self.villain_goal = input(self.villainorlack()[2] + '\n')
        self.villain_goal_bert = list(self.bert_embedding(self.villain_goal, model, tokenizer))
        self.villorlack['VillainGoal'] = self.villain_goal  # , self.villain_goal_bert

        self.villain_action = input(self.villainorlack()[3] + '\n')
        self.villain_action_bert = list(self.bert_embedding(self.villain_action, model, tokenizer))
        self.villorlack['VillainAction'] = self.villain_action  # , self.villain_action_bert

        self.object_lacked = input(self.villainorlack()[4] + '\n')
        self.object_lacked_bert = list(self.bert_embedding(self.object_lacked, model, tokenizer))
        self.villorlack['ObjectLacked'] = self.object_lacked  # , self.object_lacked_bert

        self.content_dictionary['VillainyOrLack'] = self.villorlack

        print()

        # Mediation
        print('Meanwhile, your character has been preparing for the upcoming conflict.')
        time.sleep(2)
        self.hero_preparation = input(self.mediation() + '\n')
        self.hero_preparation_bert = list(self.bert_embedding(self.hero_preparation, model, tokenizer))
        self.mediate['HeroPreparation'] = self.hero_preparation  # , self.hero_preparation_bert

        self.content_dictionary['Mediation'] = self.mediate

        print()

        # Punishment
        print('You and the other heroes fight the villain and prevail and now need to decide on a fitting punishment.')
        time.sleep(2)
        self.villain_punishment = input(self.punishment() + '\n')
        self.villain_punishment_bert = list(self.bert_embedding(self.villain_punishment, model, tokenizer))
        self.punish['VillainPunishment'] = self.villain_punishment  # , self.villain_punishment_bert

        self.content_dictionary['Punishment'] = self.punish

        print()

        # Wedding (or other positive event for the heroes)
        print('After the resolution of the conflict, it is time for you and the heroes to celebrate.')
        time.sleep(2)
        self.hero_reward = input(self.wedding() + '\n')
        self.hero_reward_bert = list(self.bert_embedding(self.hero_reward, model, tokenizer))
        self.wed['HeroReward'] = self.hero_reward  # , self.hero_reward_bert

        self.content_dictionary['Wedding'] = self.wed

        print()


        # Thanks
        time.sleep(2)
        print('Thank you. Generating story...')

        #pprint.pprint(self.content_dictionary)


    def bert_embedding(self, content, model, tokenizer):
        encoding = tokenizer.encode_plus(
            content,
            add_special_tokens=True,
            padding='max_length',
            return_attention_mask=True,
            return_token_type_ids=False,
            return_tensors='pt')

        output = model(encoding['input_ids'], encoding['attention_mask'])
        vector = output[0][0][0].detach().cpu().numpy()

        return vector

    def introduction(self):
        title = "What is the title of your story?"
        kingdom = "What kingdom is your story set in?"
        landscape = "Please describe the landscape"
        magic = "Is magic available in this universe?"
        sentiment = "What is the sentiment of this story (e.g., sad, happy, hopeful, scary, adventurous, etc.)?"

        return title, kingdom, landscape, magic, sentiment

    def absentation(self):
        goal = "What goal does your character want to achieve?"
        motivation = "Why does your character want to achieve this goal?"

        return goal, motivation

    def interdiction(self):
        danger = "What is the danger/risk of what your character wants to achieve?"
        warning = "Someone warns your character of the danger of achieving this goal. What is the advice?"
        giver = "Who gives your character this advice?"

        return danger, warning, giver

    def violate_advice(self):
        hero_action = "What does your character do to violate the advice?"

        return hero_action

    def reconaissance(self):
        recon = "What has the villain learned about your character?"

        return recon

    def villainorlack(self):
        villainy = "Does the villain do something horrible (please enter True or False)?"
        lack = "Does your character lack something (please enter True or False)?"
        villain_goal = "What is the villain's goal?"
        villain_action = "What does the villain do?"
        object_lacked = "What is your character lacking?"

        return villainy, lack, villain_goal, villain_action, object_lacked

    def mediation(self):
        hero_preparation = "How does your character prepare for the upcoming conflict with the villain (e.g., by addressing lack)?"

        return hero_preparation

    def punishment(self):
        villain_punishment = "How do your character and the other heroes punish the villain?"

        return villain_punishment

    def wedding(self):
        hero_reward = 'What positive reward/event do your character and the other heroes get in your character\'s story?'

        return hero_reward

    def characterinfo(self):
        creature = "What kind of fantasy creature is your character? (e.g., elf, orc, dwarf, human, etc.)?"
        archetype = "What archetype is your character? (e.g., hunter, wizard, warrior, etc.)?"
        companions = "How many travel companions does your character have?"

        return creature, archetype, companions


def main():
    story_initializer = StoryInitializer()


if __name__ == "__main__":
    main()
