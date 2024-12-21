from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from fake_useragent import FakeUserAgent
from bs4 import BeautifulSoup
from dotenv import dotenv_values
import datetime
import csv
import random
import re

OPTIONS = [
  "--disable-blink-features=AutomationControlled",
  #"--headless",
  "--user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36" # Instead of HeadlessChrome
]

EXPERIMENTAL = {
  "excludeSwitches": ["enable-automation"],
  "useAutomationExtension": False
}

LINKS = {
  "twitterLogin": "https://x.com/i/flow/login"
}

# Yes, that is AI-generated
KEYWORDS = [
    # Technology
    "AI", "MachineLearning", "Python", "TechNews", "CyberSecurity", "Blockchain", "Programming", "WebDevelopment", "DataScience",
    # Sports
    "Football", "Soccer", "NBA", "FIFA", "Tennis", "Olympics", "Cricket", "Formula1", "SportsNews",
    # Entertainment
    "Movies", "Netflix", "Music", "Gaming", "Hollywood", "Bollywood", "Anime", "Streaming", "CelebrityNews",
    # News and Politics
    "BreakingNews", "WorldNews", "Politics", "Elections", "ClimateChange", "Economy", "Protests", "COVID19", "GlobalEvents",
    # Lifestyle and Hobbies
    "Travel", "Foodie", "Cooking", "Photography", "Fitness", "DIY", "Gardening", "Books", "Art", "Pets",
    # Social Movements
    "SocialJustice", "MentalHealth", "Equality", "Diversity", "LGBTQ", "Feminism", "BLM", "Activism",
    # Business and Finance
    "StockMarket", "Crypto", "Investing", "Startups", "Entrepreneurship", "BusinessNews", "Economics", "NFTs",
    # Miscellaneous
    "Memes", "Funny", "LifeHacks", "Motivation", "Quotes", "Nature", "Space", "Science", "Education",
    # Specific Events
    "SuperBowl", "Oscars", "WorldCup", "MetGala", "BlackFriday", "CyberMonday", "NewYear2024", "Christmas", "Halloween"
]

class Crawler:
  def __init__(self):
    options = webdriver.ChromeOptions()
    for opt in OPTIONS:
      options.add_argument(opt)
    for (name, value) in EXPERIMENTAL.items():
      options.add_experimental_option(name, value)
    driver = webdriver.Chrome(options=options)
    driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
    driver.implicitly_wait(30.)
    self.driver = driver
    self.ua = FakeUserAgent()
    self.config = dotenv_values(".env")
    self.seen = set()
    self.dump = open(f"data/{datetime.datetime.now()}.csv", 'w', newline='')
    self.writer = csv.DictWriter(self.dump, fieldnames=['content'])
    self.count = 0
  
  def fill_form(self, xpath, value: str):
    wait = WebDriverWait(self.driver, 60)
    action = ActionChains(self.driver)
    element = wait.until(EC.element_to_be_clickable((By.XPATH, xpath)))
    action.click(on_element=element)
    for v in value:
      action.send_keys(v)
      action.pause(random.uniform(.1, .3))
    action.send_keys(Keys.ENTER)
    action.perform()
    wait.until(EC.staleness_of(element))

  def scroll_down(self):
    action = ActionChains(self.driver)
    action.scroll_by_amount(0, 500)
    action.pause(random.uniform(1., 2.))
    action.perform()

  def get_tweets(self):
    wait = WebDriverWait(self.driver, 60)
    xpath = '//*[div[@data-testid="tweetText" and @lang="en"]]'
    elements = [div.get_attribute("innerHTML") for div in wait.until(EC.presence_of_all_elements_located((By.XPATH, xpath)))]
    for element in elements:
      soup = BeautifulSoup(element, "html.parser")
      if soup.find("div", {"data-testid": "tweet-text-show-more-link"}) != None:
        continue
      result = ""
      for tag in soup.find("div").find_all(re.compile("(?:span)|(?:img)")):
        if tag.name == "span":
          result = result + tag.text
        else:
          result = result + tag["alt"]
      result = " ".join(result.split())
      if result in self.seen:
        continue 
      self.seen.add(result)
      self.writer.writerow({"content": result})
      self.dump.flush()
      self.count = self.count + 1
      
  def search(self, keyword):
    xpath = '//input[@data-testid="SearchBox_Search_Input"]'
    self.fill_form(xpath, keyword)

  def switch_agent(self):
    agent = self.ua.random
    self.driver.execute_cdp_cmd('Network.setUserAgentOverride', {"userAgent": agent})

  def twitter_login(self):
    try:
      self.driver.get(LINKS["twitterLogin"])
      self.fill_form('//input[@autocomplete="username"]', self.config["USERNAME"])
      self.fill_form('//input[@autocomplete="current-password"]', self.config["PASSWORD"])
    except:
      raise Exception("⚠️ Could not log into twitter... (run without --headless to see why)") from None

  def run(self, keywords):
    self.twitter_login()
    random.shuffle(keywords)
    for keyword in keywords:
      try:
        self.count = 0
        self.search(keyword)
        while self.count < 200:
          self.get_tweets()
          self.scroll_down()
        self.switch_agent() # Never let them know your next move
      except: 
        raise Exception(f"⚠️ Something went wrong on keyword={keyword}. Stopping... ({len(self.seen)} tweets scraped)") from None
  
  def run_no_search(self):
    self.twitter_login()
    self.count = 0
    try:
      while self.count <= 1000:
        self.get_tweets()
        self.scroll_down()
    except: 
      raise Exception(f"Elon musk more like Elon SUCKS ({len(self.seen)} tweets scraped)") from None

  def __del__(self):
    self.dump.close()
    self.driver.quit()

if __name__ == "__main__":
  crawler = Crawler()
  #crawler.run(KEYWORDS)
  crawler.run_no_search()
