print("Starting Scraper")

import json
import time

from selenium import webdriver
from selenium.webdriver.chrome.options import Options


"""
format of tree is
dict {
    tree_id: tree_id_text
    context: context text?
    first_story_block
    action_results: [act_res1, act_res2, act_res3...]
}

where each action_result's format is:
dict{
    action: action_text
    result: result_text
    action_results: [act_res1, act_res2, act_res3...]
}
"""


class Scraper:
    def __init__(self):
        chrome_options = Options()
        chrome_options.add_argument("--binary=/path/to/other/chrome/binary")
        chrome_options.add_argument("--incognito")
        chrome_options.add_argument("--window-size=1920x1080")
        exec_path = "C:\chromedriver\chromedriver"
        self.driver = webdriver.Chrome(
            chrome_options=chrome_options, executable_path=exec_path
        )
        self.max_depth = 10

        self.texts = set()

    def atEnding(self):
        try:
            img = self.driver.find_element_by_class_name("end-of-story")
            return True
        except:
            return False

    def GoToURL(self, url):
        self.texts = set()
        self.driver.get(url)
        time.sleep(0.5)

    def GetText(self):
        text = self.driver.find_element_by_class_name("description").text
        return text

    def GetLinks(self):
        links = []
        list = self.driver.find_elements_by_css_selector("a")
        if list:
            for link in list:
                if link.get_attribute("href") and ("story/choice" in link.get_attribute("href")):
                    links.append(link)
        # print(links)
        return links

    def GoBack(self):
        # self.GetLinks()[0].click()
        self.driver.back()
        time.sleep(0.2)

    def ClickAction(self, links, action_num):
        links[action_num].click()
        time.sleep(0.2)

    def GetActions(self):
        # print(self.GetLinks())
        return [link.text for link in self.GetLinks()]

    def NumActions(self):
        return len(self.GetLinks())

    def BuildTreeHelper(self, parent_story, action_num, depth, old_actions):
        depth += 1
        action_result = {}

        action = old_actions[action_num]
        print("Action is ", repr(action))
        action_result["action"] = action

        links = self.GetLinks()
        if action_num >= len(links):
            print("action index over limit")
            return None

        try:
            self.ClickAction(links, action_num)
        except:
            print("click failed")
            return None
        result = self.GetText()
        if result == parent_story or result in self.texts:
            print("we've been here already")
            self.GoBack()
            return None

        self.texts.add(result)
        print(len(self.texts))

        action_result["result"] = result

        actions = self.GetActions()
        action_result["action_results"] = []

        # print(actions)
        if len(actions) == 0:
            return action_result

        for i, action in enumerate(actions):
            if not self.atEnding():
                sub_action_result = self.BuildTreeHelper(result, i, depth, actions)
                if action_result is not None:
                    action_result["action_results"].append(sub_action_result)

        try:
            self.GoBack()
        except:
            print("We hit that weird comment screen bug")
        time.sleep(0.2)
        return action_result

    def BuildStoryTree(self, url):
        scraper.GoToURL(url)
        text = scraper.GetText()
        actions = self.GetActions()
        story_dict = {}
        story_dict["tree_id"] = url
        story_dict["context"] = ""
        story_dict["first_story_block"] = text
        story_dict["action_results"] = []

        for i, action in enumerate(actions):
            if not self.atEnding():
                action_result = self.BuildTreeHelper(text, i, 0, actions)
                if action_result is not None:
                    story_dict["action_results"].append(action_result)
            else:
                print("done")

        return story_dict


def save_tree(tree, filename):
    with open(filename, "w") as fp:
        json.dump(tree, fp)


scraper = Scraper()

urls = [
    "https://infinite-story.com/story/room.php?id=87587",
    "https://infinite-story.com/story/room.php?id=41162",
    "https://infinite-story.com/story/room.php?id=31984",
    "https://infinite-story.com/story/room.php?id=56017",
    "https://infinite-story.com/story/room.php?id=94415",
    "https://infinite-story.com/story/room.php?id=33470",
    "https://infinite-story.com/story/room.php?id=36382",
    "https://infinite-story.com/story/room.php?id=28071",
    "https://infinite-story.com/story/room.php?id=46933",
    "https://infinite-story.com/story/room.php?id=24731",
    "https://infinite-story.com/story/room.php?id=22276",

]

for i in range(2, len(urls)):
    print("****** Extracting Adventure ", urls[i], " ***********", "index: ", i)
    tree = scraper.BuildStoryTree(urls[i])
    print(type(tree))
    save_tree(tree, "infinite_stories/story" + str(1 + i) + ".json")

print("done")
