#!/usr/bin/env python3
import csv
import os

from dotenv import load_dotenv
load_dotenv()

import sqlite3
from gtts import gTTS
from playwright.sync_api import sync_playwright
import cv2

browser_type = os.getenv('AITA_BROWSER', 'chromium')
assert browser_type in ('chromium', 'firefox', 'webkit')

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

removed_im_path = os.path.join(inference_folder, 'removed.png')
removed_im = cv2.imread(removed_im_path)
deleted_im_path = os.path.join(inference_folder, 'deleted.png')
deleted_im = cv2.imread(deleted_im_path)
removed_im_threshold = 0.8

def was_removed(im):
  for other_im in [removed_im, deleted_im]:
    if (cv2.matchTemplate(im, other_im, cv2.TM_CCOEFF_NORMED) >= removed_im_threshold).any():
      return True
  return False

def get_videos(posts, limit=-1):
  with sync_playwright() as p:
    browser = p[browser_type].launch()
    try:
      cur = sqlite3.connect(played_dht)
      page = browser.new_page()
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
          page.goto(url)
          page.mouse.wheel(0, 224)
          page.screenshot(path=im_path)

          # Test to see if the page was removed using OpenCV
          im = cv2.imread(im_path)
          if was_removed(im):
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
