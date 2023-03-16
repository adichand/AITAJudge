#!/usr/bin/env python3
import asyncio
from contextlib import closing
import csv
import os
import io
from xml.dom import minidom

from dotenv import load_dotenv
load_dotenv()

import sqlite3
import gtts
from selenium import webdriver
import PIL.Image
import ffmpeg

browsers = {
  'chrome': webdriver.Chrome,
  'firefox': webdriver.Firefox,
}
browser_options = {
  'chrome': webdriver.chrome.options.Options(),
  'firefox': webdriver.FirefoxOptions(),
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

firefox = browser_options['firefox']
firefox.add_argument("--headless")
firefox.add_argument("--width=1280")
firefox.add_argument("--height=720")


browser_type = os.getenv('AITA_BROWSER', 'chrome')
assert browser_type in ('chrome', 'firefox')

inference_folder = os.path.dirname(os.path.abspath(__file__))

played_dht = os.path.join(inference_folder, 'played.db')
if not os.path.exists(played_dht):
  raise Exception("You need to run 'clean_csvs.py' in the dataset folder first to get the Reddit posts.")

removed_im_path = os.path.join(inference_folder, 'removed.svg')
removed_im = minidom.parse(removed_im_path).getElementsByTagName('path')[0].getAttribute('d')
deleted_im_path = os.path.join(inference_folder, 'deleted.svg')
deleted_im = minidom.parse(deleted_im_path).getElementsByTagName('path')[0].getAttribute('d')
awaiting_im_path = os.path.join(inference_folder, 'awaiting.svg')
awaiting_im = minidom.parse(deleted_im_path).getElementsByTagName('path')[0].getAttribute('d')

use_named_pipes = os.getenv('AITA_NAMED_PIPES', '') == '1'

gtts.tokenizer.symbols.SUB_PAIRS.extend((
  ('YTA', "You're the asshole"),
  ('NTA', "Not the asshole"),
  ('ESH', "Everyone sucks here"),
  ('NAH', "No assholes here"),
  ('AITA', "Am I the asshole"),
  ('WIBTA', "Would I be the asshole"),
  # ('tl;dr', "Tea El Dee Are"),
))

async def tts(selftext, comment, redd_id):
  # Create a named pipe if on Unix
  # This will make the processing faster
  if use_named_pipes:
    try:
      os.mkfifo(f'tmp_0_{redd_id}.mp3')
      os.mkfifo(f'{redd_id}.mp3')
    except OSError as oe:
      if oe.errno != errno.EEXIST:
        raise

  segments = []

  # The actual text
  with open(f'tmp_0_{redd_id}.mp3', 'wb') as f:
    tts = gtts.gTTS(selftext, lang='en')
    tts.write_to_fp(f)

  audio = ffmpeg.input(f'tmp_0_{redd_id}.mp3')
  segments.append(audio)

  # Silence
  audio = ffmpeg.input(
    'anullsrc=channel_layout=mono:sample_rate=24000',
    format='lavfi',
    t=1 # how long you want the silence to be in seconds
  )
  segments.append(audio)

  # The comment
  audio = ffmpeg.input(f'pipe:')
  segments.append(audio)

  combined = ffmpeg.concat(*segments, v=0, a=1)
  process = combined.output(f'{redd_id}.mp3').global_args('-loglevel', 'error') \
    .run_async(pipe_stdin=True)

  tts = gtts.gTTS(comment, lang='en')
  tts.write_to_fp(process.stdin)
  process.stdin.close()
  process.wait()

  os.remove(f'tmp_0_{redd_id}.mp3')


async def get_videos(posts, model, limit=-1):
  new_snips = 0
  tts_promises = []

  if limit == 0: return 0

  with browsers[browser_type](options=browser_options[browser_type]) as driver, \
      closing(sqlite3.connect(played_dht)) as con:
    dx, dy = driver.execute_script("var w=window; return [w.outerWidth - w.innerWidth, w.outerHeight - w.innerHeight];")
    if dx > 0 and dy > 0:
      driver.set_window_size(1280 + dx, 720 + dy)

    driver.get('https://www.reddit.com/r/AmItheAsshole') # Load some cookies?

    for post in posts:
      selftext = post['selftext']
      url = post['url']
      redd_id = post['id']
      if con.execute("SELECT * FROM PostsPlayed WHERE PostId == ?", (redd_id,)).fetchone(): # will be none by default
        continue
      im_path = f'{redd_id}.png'
      no_img = not os.path.exists(im_path)

      if no_img:
        print(f"capture {redd_id}")
        driver.get(url)

        # Test to see if the page was removed
        if any(
          tag.get_attribute('d') in (removed_im, deleted_im, awaiting_im)
          for tag in driver.find_elements(webdriver.common.by.By.TAG_NAME, 'path')
        ):
          con.execute('INSERT INTO PostsPlayed VALUES (?,?,?)', (redd_id, True, 0))
          con.commit()
          print(f'{redd_id} was removed')
          continue

      # Start downloading the mp3 before the screenshot is taken to save
      # a little bit of time
      if not os.path.exists(f'{redd_id}.mp3'):
        print(f"tts {redd_id}")
        comment = model(post)
        if comment is None:
          con.execute('INSERT INTO PostsPlayed VALUES (?,?,?)', (redd_id, True, 0))
          con.commit()
          print(f'no AI comment {redd_id}')
          continue
        tts_promises.append(tts(selftext, comment, redd_id))
        new_snips += 1

      # Take the screenshot if the photo isn't there
      if no_img:
        (
          webdriver.ActionChains(driver)
            .scroll_by_amount(0, 224)
            .perform()
        )

        png = driver.get_screenshot_as_png()

        if use_named_pipes:
          try:
            os.mkfifo(f'{redd_id}.png')
          except OSError as oe:
            if oe.errno != errno.EEXIST:
              raise

        # Reduce resolution so my poor MacBook Air will be happy.
        im = PIL.Image.open(io.BytesIO(png))
        im = im.resize((1280, 720))
        im.save(im_path)

      if new_snips == limit:
        break

  for promise in tts_promises:
    await promise

  return new_snips

def load_model(model_path):
  cwd = os.getcwd()
  os.chdir(os.path.dirname(__file__))
  import nnsave
  os.chdir(cwd)

  model_path = os.path.join(os.getcwd(), model_path)
  models_folder = os.path.join(os.path.dirname(__file__), '..')
  with nnsave.PackageSandbox(models_folder) as sand:
    model = sand.load_pickle(os.path.relpath(model_path, models_folder))
    from models.wrappers import Commenter, _fallback
    if not isinstance(model, Commenter):
      print("Model not Commenter. Rewrapping.")
      model = _fallback(model)
    return model
  # os.chdir(os.path.join(os.path.dirname(__file__), '../models'))
  # with open(model_path, 'rb') as f:
  #   model = pickle.load(f)
  # os.chdir(cwd)
  # return model

def load_posts(path):
  if path is None:
    # SQL database
    with closing(sqlite3.connect(played_dht)) as con:
      con.row_factory = sqlite3.Row
      while True:
        yield from con.execute("""
        SELECT Posts.* FROM Posts
        LEFT JOIN PostsPlayed ON PostsPlayed.PostId == Posts.id
        WHERE PostsPlayed.PostId IS NULL
        """).fetchall()
  else:
    # CSV
    if not os.path.isabs(path):
      path = os.path.join('..', path)
    with open(path, 'r') as f:
      yield from csv.DictReader(f)

if __name__ == '__main__':
  import argparse
  p = argparse.ArgumentParser(description="Download audio snippets from gTTS")
  p.add_argument("-path", help="path of the posts.csv from the dataset", default=None) # '../dataset/posts_inference.csv'
  p.add_argument("-model", help="path of the model.pkl file", default='../models/sklearn_models/saved_models/nn_regressor_nrows=20000_generic.pkl')
  p.add_argument("-limit", help="the number of audio clips to download", type=int, default=-1)
  P = p.parse_args()

  model_path = os.path.join(os.path.dirname(__file__), P.model)
  model = load_model(model_path)

  os.chdir(os.path.join(os.path.dirname(__file__), 'streams'))
  asyncio.run(get_videos(load_posts(P.path), model, P.limit))
