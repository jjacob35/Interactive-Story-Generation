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
        #chrome_options.add_argument("--binary=/path/to/other/chrome/binary")
        chrome_options.add_argument("--incognito")
        chrome_options.add_argument("--window-size=1920x1080")
        exec_path = "C:/Users/jmian/OneDrive/Documents/GT/MSCS/Fall_2020/CS8803_DLT/Projects/ChromeDriver/chromedriver.exe"
        self.driver = webdriver.Chrome(
            options=chrome_options, executable_path=exec_path
        )
        self.max_depth = 10
        self.end_actions = {
            "End Game and Leave Comments",
            "Click here to End the Game and Leave Comments",
            "See How Well You Did (you can still back-page afterwards if you like)",
            "You have died.",
            "You have died",
            "Epilogue",
            "Save Game",
            "Your quest might have been more successful...",
            "5 - not the best, certainly not the worst",
            "The End! (leave comments on game)",
            "6 - it's worth every cent",
            "You do not survive the journey to California",
            "Quit the game.",
            "7 - even better than Reeses' Cups®",
            "8 - it will bring you enlightenment",
            "End of game! Leave a comment!",
            "Better luck next time",
            "click here to continue",
            "Rating And Leaving Comments",
            "You do not survive your journey to California",
            "Your Outlaw Career has come to an end",
            "Thank you for taking the time to read my story",
            "You have no further part in the story, End Game and Leave Comments",
            "",
            "You play no further part in this story. End Game and Leave Comments",
            "drivers",
            "Alas, poor Yorick, they slew you well",
            "My heart bleeds for you",
            "To End the Game and Leave Comments click here",
            "Call it a day",
            "Check the voicemail.",
            "reset",
            "There's nothing you can do anymore...it's over.",
            "To Be Continued...",
            "Thanks again for taking the time to read this",
            "If you just want to escape this endless story you can do that by clicking here",
            "Boo Hoo Hoo",
            "End.",
            "Pick up some money real quick",
            "",
            "Well you did live a decent amount of time in the Army",
            "End Game",
            "You have survived the Donner Party's journey to California!",
        }
        self.texts = set()

    def GoToURL(self, url):
        self.texts = set()
        self.driver.get(url)
        time.sleep(0.5)

    def GetText(self):
        div_elements = self.driver.find_elements_by_css_selector("div")
        text = div_elements[3].text
        return text

    def GetLinks(self):
        return self.driver.find_elements_by_css_selector("a")

    def GoBack(self):
        self.GetLinks()[0].click()
        time.sleep(0.2)

    def ClickAction(self, links, action_num):
        links[action_num + 4].click()
        time.sleep(0.2)

    def GetActions(self):
        return [link.text for link in self.GetLinks()[4:]]

    def NumActions(self):
        return len(self.GetLinks()) - 4

    def BuildTreeHelper(self, parent_story, action_num, depth, old_actions):
        depth += 1
        action_result = {}

        action = old_actions[action_num]
        print("Action is ", repr(action))
        action_result["action"] = action

        links = self.GetLinks()
        if action_num + 4 >= len(links):
            return None

        self.ClickAction(links, action_num)
        result = self.GetText()
        if result == parent_story or result in self.texts:
            self.GoBack()
            return None

        self.texts.add(result)
        print(len(self.texts))

        action_result["result"] = result

        actions = self.GetActions()
        action_result["action_results"] = []

        for i, action in enumerate(actions):
            if actions[i] not in self.end_actions:
                sub_action_result = self.BuildTreeHelper(result, i, depth, actions)
                if action_result is not None:
                    action_result["action_results"].append(sub_action_result)

        self.GoBack()
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
            if action not in self.end_actions:
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
"http://chooseyourstory.com/story/viewer/default.aspx?StoryId=10638",
"http://chooseyourstory.com/story/viewer/default.aspx?StoryId=11246",
"http://chooseyourstory.com/story/viewer/default.aspx?StoryId=54639",
"http://chooseyourstory.com/story/viewer/default.aspx?StoryId=7397",
"http://chooseyourstory.com/story/viewer/default.aspx?StoryId=8041",
"http://chooseyourstory.com/story/viewer/default.aspx?StoryId=11545",
"http://chooseyourstory.com/story/viewer/default.aspx?StoryId=7393",
"http://chooseyourstory.com/story/viewer/default.aspx?StoryId=13875",
"http://chooseyourstory.com/story/viewer/default.aspx?StoryId=37696",
"http://chooseyourstory.com/story/viewer/default.aspx?StoryId=31013",
"http://chooseyourstory.com/story/viewer/default.aspx?StoryId=45375",
"http://chooseyourstory.com/story/viewer/default.aspx?StoryId=41698",
"http://chooseyourstory.com/story/viewer/default.aspx?StoryId=10634",
"http://chooseyourstory.com/story/viewer/default.aspx?StoryId=42204",
"http://chooseyourstory.com/story/viewer/default.aspx?StoryId=6823",
"http://chooseyourstory.com/story/viewer/default.aspx?StoryId=18988",
"http://chooseyourstory.com/story/viewer/default.aspx?StoryId=10359",
"http://chooseyourstory.com/story/viewer/default.aspx?StoryId=5466",
"http://chooseyourstory.com/story/viewer/default.aspx?StoryId=28030",
"http://chooseyourstory.com/story/viewer/default.aspx?StoryId=56515",
"http://chooseyourstory.com/story/viewer/default.aspx?StoryId=7480",
"http://chooseyourstory.com/story/viewer/default.aspx?StoryId=11274",
"http://chooseyourstory.com/story/viewer/default.aspx?StoryId=53134",
"http://chooseyourstory.com/story/viewer/default.aspx?StoryId=17306",
"http://chooseyourstory.com/story/viewer/default.aspx?StoryId=470",
"http://chooseyourstory.com/story/viewer/default.aspx?StoryId=23928",
"http://chooseyourstory.com/story/viewer/default.aspx?StoryId=10183",
"http://chooseyourstory.com/story/viewer/default.aspx?StoryId=45866",
"http://chooseyourstory.com/story/viewer/default.aspx?StoryId=60232",
"http://chooseyourstory.com/story/viewer/default.aspx?StoryId=6376",
"http://chooseyourstory.com/story/viewer/default.aspx?StoryId=36791",
"http://chooseyourstory.com/story/viewer/default.aspx?StoryId=60128",
"http://chooseyourstory.com/story/viewer/default.aspx?StoryId=52961",
"http://chooseyourstory.com/story/viewer/default.aspx?StoryId=54011",
"http://chooseyourstory.com/story/viewer/default.aspx?StoryId=34838",
"http://chooseyourstory.com/story/viewer/default.aspx?StoryId=13349",
"http://chooseyourstory.com/story/viewer/default.aspx?StoryId=8038",
"http://chooseyourstory.com/story/viewer/default.aspx?StoryId=56742",
"http://chooseyourstory.com/story/viewer/default.aspx?StoryId=48393",
"http://chooseyourstory.com/story/viewer/default.aspx?StoryId=53356",
"http://chooseyourstory.com/story/viewer/default.aspx?StoryId=10872",
"http://chooseyourstory.com/story/viewer/default.aspx?StoryId=43910",
"http://chooseyourstory.com/story/viewer/default.aspx?StoryId=53837",
"http://chooseyourstory.com/story/viewer/default.aspx?StoryId=8098",
"http://chooseyourstory.com/story/viewer/default.aspx?StoryId=55043",
"http://chooseyourstory.com/story/viewer/default.aspx?StoryId=28838",
"http://chooseyourstory.com/story/viewer/default.aspx?StoryId=11906",
"http://chooseyourstory.com/story/viewer/default.aspx?StoryId=8040",
"http://chooseyourstory.com/story/viewer/default.aspx?StoryId=2280",
"http://chooseyourstory.com/story/viewer/default.aspx?StoryId=31014",
"http://chooseyourstory.com/story/viewer/default.aspx?StoryId=43744",
"http://chooseyourstory.com/story/viewer/default.aspx?StoryId=44543",
"http://chooseyourstory.com/story/viewer/default.aspx?StoryId=56753",
"http://chooseyourstory.com/story/viewer/default.aspx?StoryId=36594",
"http://chooseyourstory.com/story/viewer/default.aspx?StoryId=8035",
"http://chooseyourstory.com/story/viewer/default.aspx?StoryId=10524",
"http://chooseyourstory.com/story/viewer/default.aspx?StoryId=14899",
"http://chooseyourstory.com/story/viewer/default.aspx?StoryId=9361",
"http://chooseyourstory.com/story/viewer/default.aspx?StoryId=49642",
"http://chooseyourstory.com/story/viewer/default.aspx?StoryId=43573",
"http://chooseyourstory.com/story/viewer/default.aspx?StoryId=38025",
"http://chooseyourstory.com/story/viewer/default.aspx?StoryId=7567",
"http://chooseyourstory.com/story/viewer/default.aspx?StoryId=60747",
"http://chooseyourstory.com/story/viewer/default.aspx?StoryId=31353",
"http://chooseyourstory.com/story/viewer/default.aspx?StoryId=56501",
"http://chooseyourstory.com/story/viewer/default.aspx?StoryId=38542",
"http://chooseyourstory.com/story/viewer/default.aspx?StoryId=43993",
"http://chooseyourstory.com/story/viewer/default.aspx?StoryId=1153",
"http://chooseyourstory.com/story/viewer/default.aspx?StoryId=24743",
"http://chooseyourstory.com/story/viewer/default.aspx?StoryId=57114",
"http://chooseyourstory.com/story/viewer/default.aspx?StoryId=52887",
"http://chooseyourstory.com/story/viewer/default.aspx?StoryId=21879",
"http://chooseyourstory.com/story/viewer/default.aspx?StoryId=16489",
"http://chooseyourstory.com/story/viewer/default.aspx?StoryId=53186",
"http://chooseyourstory.com/story/viewer/default.aspx?StoryId=34849",
"http://chooseyourstory.com/story/viewer/default.aspx?StoryId=26752",
"http://chooseyourstory.com/story/viewer/default.aspx?StoryId=7094",
"http://chooseyourstory.com/story/viewer/default.aspx?StoryId=8557",
"http://chooseyourstory.com/story/viewer/default.aspx?StoryId=45225",
"http://chooseyourstory.com/story/viewer/default.aspx?StoryId=4720",
"http://chooseyourstory.com/story/viewer/default.aspx?StoryId=51926",
"http://chooseyourstory.com/story/viewer/default.aspx?StoryId=27234",
"http://chooseyourstory.com/story/viewer/default.aspx?StoryId=60772",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=24540",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=25370",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=12487",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=50303",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=27800",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=60232",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=14237",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=14188",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=11951",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=34956",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=61398",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=[]",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=34138",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=52067",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=27880",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=60671",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=61367",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=10489",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=25548",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=45379",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=5587",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=13922",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=25584",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=17576",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=1118",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=52499",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=10721",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=10808",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=57778",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=10512",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=33366",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=30352",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=30321",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=8041",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=29206",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=24535",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=10056",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=19882",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=61827",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=12792",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=19780",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=28962",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=33454",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=10638",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=47074",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=16107",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=38757",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=63449",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=40489",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=46107",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=48393",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=11393",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=18504",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=42701",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=31243",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=8038",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=8562",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=10634",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=16176",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=11178",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=9444",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=38431",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=38774",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=39055",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=39595",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=45493",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=36791",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=17306",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=33462",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=33496",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=61583",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=18722",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=64095",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=63197",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=7397",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=9978",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=9958",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=60128",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=47509",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=45866",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=9095",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=10226",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=23179",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=29597",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=40947",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=9789",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=14304",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=13349",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=52961",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=54639",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=9687",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=38932",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=18496",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=64475",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=51334",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=58046",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=60858",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=45480",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=40688",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=50613",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=19493",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=18645",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=27163",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=48362",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=5388",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=17620",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=16577",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=45266",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=19778",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=51231",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=23928",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=11274",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=38525",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=36085",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=52108",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=36462",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=61587",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=37416",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=25611",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=26777",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=2272",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=2302",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=47568",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=63542",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=58502",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=58936",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=54011",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=36247",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=60875",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=53134",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=53514",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=54768",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=34751",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=37696",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=7513",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=16710",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=10183",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=60258",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=8875",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=15494",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=21334",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=57746",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=15986",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=30423",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=38507",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=60539",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=38691",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=21887",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=17104",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=40781",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=35389",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=58170",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=8356",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=34838",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=59583",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=58487",
"https://chooseyourstory.com/story/viewer/default.aspx?StoryId=48438"

]

for i in range(0, len(urls)):
    print("****** Extracting Adventure ", urls[i], " ***********")
    tree = scraper.BuildStoryTree(urls[i])
    save_tree(tree, "stories/story" + str(41 + i) + ".json")

print("done")