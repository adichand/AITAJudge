#!/usr/bin/env python3
import csv
import os

from dotenv import load_dotenv
load_dotenv()

import sqlite3
from gtts import gTTS
from selenium import webdriver
from xml.dom import minidom

browsers = {
  'chrome': webdriver.Chrome,
}
browser_options = {
  'chrome': webdriver.chrome.options.Options(),
}
chrome = browser_options['chrome']
chrome.add_argument("--headless")
# chrome.add_argument("--window-size=640,360")
chrome.add_argument("--enable-use-zoom-for-dsf=false")
chrome.add_argument("--hide-scrollbars")
chrome.add_argument("--mute-audio")
chrome.add_argument("--disable-default-apps")
chrome.add_argument("--disable-extensions")
chrome.add_argument("--disable-component-update")
chrome.add_argument("--disable-back-forward-cache")
chrome.add_argument("--disable-backgrounding-occluded-windows")



browser_type = os.getenv('AITA_BROWSER', 'chrome')
assert browser_type in ('chrome',)

inference_folder = os.path.dirname(__file__)

played_dht = os.path.join(inference_folder, 'played.db')
if not os.path.exists(played_dht):
  # Essentially touch the file
  cur = sqlite3.connect(played_dht)

  cur.execute("""
  CREATE TABLE Posts
  (
    PostId CHAR(8) NOT NULL,
    Removed BOOL,
    Played INT,
    PRIMARY KEY(PostId)
  )
  """)

  cur.commit()
  cur.close()

removed_im_path = os.path.join(inference_folder, 'removed.svg')
removed_im = minidom.parse(removed_im_path).getElementsByTagName('path')[0].getAttribute('d')
deleted_im_path = os.path.join(inference_folder, 'deleted.svg')
deleted_im = minidom.parse(deleted_im_path).getElementsByTagName('path')[0].getAttribute('d')

def get_videos(posts, limit=-1):
  with browsers[browser_type](options=browser_options[browser_type]) as driver:
    try:
      cur = sqlite3.connect(played_dht)
      new_snips = 0
      for post in posts:
        selftext = post['selftext']
        url = post['url']
        redd_id = post['id']
        if cur.execute("SELECT * FROM Posts WHERE PostId == ?", (redd_id,)).fetchone(): # will be none by default
          continue
        im_path = f'{redd_id}.png'
        if not os.path.exists(im_path):
          print(f"capture {redd_id}")
          driver.get(url)
          (
            webdriver.ActionChains(driver)
              .scroll_by_amount(0, 224)
              .perform()
          )
          driver.save_screenshot(im_path)

          # Test to see if the page was removed
          if any(
            tag.get_attribute('d') in (removed_im, deleted_im)
            for tag in driver.find_elements(webdriver.common.by.By.TAG_NAME, 'path')
          ):
            cur.execute('INSERT INTO Posts VALUES (?,?,?)', (redd_id, True, 0))
            cur.commit()
            print(f'{redd_id} was removed')
            continue

        if not os.path.exists(f'{redd_id}.mp3'):
          print(f"tts {redd_id}")
          tts = gTTS(selftext, lang='en')
          with open(f'{redd_id}.mp3', 'wb') as f:
            tts.write_to_fp(f)
          new_snips += 1
        if new_snips == limit:
          break
    finally:
      cur.close()


if __name__ == '__main__':
  import argparse
  p = argparse.ArgumentParser(description="Download audio snippets ")
  p.add_argument("-path", help="path of the posts.csv from the dataset", default='../dataset/posts_inference.csv')
  p.add_argument("-limit", help="the number of audio clips to download", type=int, default=-1)
  P = p.parse_args()

  os.chdir(os.path.join(os.path.dirname(__file__), 'streams'))
  with open(os.path.join('..', P.path), 'r') as f:
    get_videos(csv.DictReader(f), P.limit)
