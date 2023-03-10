#!/usr/bin/env python3
import asyncio
import csv
import os
import io
from xml.dom import minidom

from dotenv import load_dotenv
load_dotenv()

import sqlite3
from gtts import gTTS
from selenium import webdriver
import PIL.Image

browsers = {
  'chrome': webdriver.Chrome,
}
browser_options = {
  'chrome': webdriver.chrome.options.Options(),
}
chrome = browser_options['chrome']
chrome.add_argument("--headless")
chrome.add_argument("--window-size=1280,720")
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

async def tts(selftext, dest):
  tts = gTTS(selftext, lang='en')
  with open(dest, 'wb') as f:
    tts.write_to_fp(f)

async def get_videos(posts, limit=-1):
  new_snips = 0
  tts_promises = []

  with browsers[browser_type](options=browser_options[browser_type]) as driver:
    dx, dy = driver.execute_script("var w=window; return [w.outerWidth - w.innerWidth, w.outerHeight - w.innerHeight];")
    if dx > 0 and dy > 0:
      driver.set_window_size(1280 + dx, 720 + dy)
    try:
      cur = sqlite3.connect(played_dht)
      for post in posts:
        selftext = post['selftext']
        url = post['url']
        redd_id = post['id']
        if cur.execute("SELECT * FROM Posts WHERE PostId == ?", (redd_id,)).fetchone(): # will be none by default
          continue
        im_path = f'{redd_id}.png'
        no_img = not os.path.exists(im_path)

        if no_img:
          print(f"capture {redd_id}")
          driver.get(url)

          # Test to see if the page was removed
          if any(
            tag.get_attribute('d') in (removed_im, deleted_im)
            for tag in driver.find_elements(webdriver.common.by.By.TAG_NAME, 'path')
          ):
            cur.execute('INSERT INTO Posts VALUES (?,?,?)', (redd_id, True, 0))
            cur.commit()
            print(f'{redd_id} was removed')
            continue

        # Start downloading the mp3 before the screenshot is taken to save
        # a little bit of time
        if not os.path.exists(f'{redd_id}.mp3'):
          print(f"tts {redd_id}")
          tts_promises.append(tts(selftext, f'{redd_id}.mp3'))
          new_snips += 1

        # Take the screenshot if the photo isn't there
        if no_img:
          (
            webdriver.ActionChains(driver)
              .scroll_by_amount(0, 224)
              .perform()
          )

          png = driver.get_screenshot_as_png()

          # Reduce resolution so my poor MacBook Air will be happy.
          im = PIL.Image.open(io.BytesIO(png))
          im = im.resize((1280, 720))
          im.save(im_path)

        if new_snips == limit:
          break
    finally:
      cur.close()

  for promise in tts_promises:
    await promise

  return new_snips


if __name__ == '__main__':
  import argparse
  p = argparse.ArgumentParser(description="Download audio snippets from gTTS")
  p.add_argument("-path", help="path of the posts.csv from the dataset", default='../dataset/posts_inference.csv')
  p.add_argument("-limit", help="the number of audio clips to download", type=int, default=-1)
  P = p.parse_args()

  os.chdir(os.path.join(os.path.dirname(__file__), 'streams'))
  with open(os.path.join('..', P.path), 'r') as f:
    asyncio.run(get_videos(csv.DictReader(f), P.limit))
