#!/usr/bin/env python3
import csv
import os

from dotenv import load_dotenv
load_dotenv()

import diskhash
from gtts import gTTS
from playwright.sync_api import sync_playwright

browser_type = os.getenv('AITA_BROWSER', 'chromium')
assert browser_type in ('chromium', 'firefox', 'webkit')


played_dht = os.path.join(os.path.dirname(__file__), 'played.db')
if not os.path.exists(played_dht):
  # Essentially touch the file
  diskhash.StructHash(played_dht, 7, '7s', 'rw')

def get_videos(posts, limit=-1):
  tb = diskhash.StructHash(played_dht, 7, '7s', 'r')
  with sync_playwright() as p:
    browser = p[browser_type].launch()
    page = browser.new_page()
    new_snips = 0
    for post in posts:
      selftext = post['selftext']
      url = post['url']
      redd_id = post['id']
      if tb.lookup(redd_id): # will be none by default
        continue
      if not os.path.exists(f'{redd_id}.png'):
        print(f"capture {redd_id}")
        page.goto(url)
        page.mouse.wheel(0, 224)
        page.screenshot(path=f'{redd_id}.png')
      if not os.path.exists(f'{redd_id}.mp3'):
        print(f"tts {redd_id}")
        tts = gTTS(selftext, lang='en')
        with open(f'{redd_id}.mp3', 'wb') as f:
          tts.write_to_fp(f)
        new_snips += 1
      if new_snips == limit:
        break
    browser.close()


if __name__ == '__main__':
  import argparse
  p = argparse.ArgumentParser(description="Download audio snippets ")
  p.add_argument("-path", help="path of the posts.csv from the dataset", default='../dataset/posts_inference.csv')
  p.add_argument("-limit", help="the number of audio clips to download", type=int, default=-1)
  P = p.parse_args()

  os.chdir(os.path.join(os.path.dirname(__file__), 'streams'))
  with open(os.path.join('..', P.path), 'r') as f:
    get_videos(csv.DictReader(f), P.limit)
