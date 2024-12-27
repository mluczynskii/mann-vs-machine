from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from bs4 import BeautifulSoup
from dotenv import dotenv_values
import traceback
import datetime
import csv
import random
import re

OPTIONS = [
  "--disable-blink-features=AutomationControlled",
  "--user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36" # Instead of HeadlessChrome
]

EXPERIMENTAL = {
  "excludeSwitches": ["enable-automation"],
  "useAutomationExtension": False
}

LINKS = {
  "twitterLogin": "https://x.com/i/flow/login"
}

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

PROMOTED="Promowane"

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
    self.config = dotenv_values(".env")
    self.seen = set()
    self.dump = open(f"data/{datetime.datetime.now()}.csv", 'w', newline='')
    self.writer = csv.DictWriter(self.dump, fieldnames=['content'])
    self.logged_in = False
  
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

  def scroll_down(self):
    action = ActionChains(self.driver)
    action.scroll_by_amount(0, round(random.uniform(500, 1000)))
    action.pause(random.uniform(1, 2))
    action.perform()

  @staticmethod
  def is_valid(soup):
    if soup.find("div", {"data-testid": "tweet-text-show-more-link"}):
      return False 
    if soup.find("span", string=lambda txt: txt == PROMOTED):
      return False
    if not soup.find("div", {"lang": "en"}):
      return False
    return True

  def get_tweets(self):
    wait = WebDriverWait(self.driver, 60)
    tweet_container = '//article[@data-testid="tweet"]'
    try:
      containers = [div.get_attribute("innerHTML") for div in wait.until(EC.presence_of_all_elements_located((By.XPATH, tweet_container)))]
    except:
      return 0
    else:  
      scraped = 0
      for html in containers:
        soup = BeautifulSoup(html, "html.parser")
        tweet = soup.find("div", {"data-testid": "tweetText"})
        if not Crawler.is_valid(soup) or not tweet:
          continue
        result = ""
        for tag in tweet.find_all(re.compile("(?:span)|(?:img)")):
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
        scraped = scraped + 1
      return scraped
      
  def search(self, keyword):
    action = ActionChains(self.driver)
    wait = WebDriverWait(self.driver, 60)
    maxlen = len(max(KEYWORDS, key=len))
    search_bar = '//input[@data-testid="SearchBox_Search_Input"]'
    element = wait.until(EC.element_to_be_clickable((By.XPATH, search_bar)))
    action.click(on_element=element)
    action.send_keys(Keys.BACKSPACE * maxlen)
    action.send_keys(keyword + Keys.ENTER)
    action.perform()

  def twitter_login(self):
    self.driver.get("https://x.com/i/flow/login")
    self.fill_form('//input[@autocomplete="username"]', self.config["USERNAME"])
    self.fill_form('//input[@autocomplete="current-password"]', self.config["PASSWORD"])
    self.logged_in = True
    
  def run(self):
    if not self.logged_in:
      self.twitter_login()
    global_strikes = 0
    while True:
      if global_strikes >= 10:
        break
      keyword = random.choice(KEYWORDS)
      self.search(keyword)
      bound = random.uniform(50, 100)
      scraped = 0
      strikes = 0
      while scraped <= bound:
        new = self.get_tweets()
        strikes = strikes + 1 if not new else 0
        if strikes >= 10:
          global_strikes = global_strikes + 1
          break
        scraped = scraped + new
        self.scroll_down()
      print(f"Scraped {scraped} tweets using keyword '{keyword}'")
    self.run_no_search()
  
  def run_no_search(self):
    if not self.logged_in:
      self.twitter_login()
    self.driver.get("https://x.com/home")
    scraped = 0
    while scraped <= 1000:
      scraped = scraped + self.get_tweets()
      self.scroll_down()

  def __del__(self):
    self.dump.close()
    self.driver.quit()

if __name__ == "__main__":
  crawler = Crawler()
  try:
    crawler.run()
  except:
    traceback.print_exc()
  finally:
    print(f'Total = {len(crawler.seen)} tweets scraped')
