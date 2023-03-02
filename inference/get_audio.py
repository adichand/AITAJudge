import os

from dotenv import load_dotenv
load_dotenv()

import diskhash
from gtts import gTTS
from moviepy.editor import AudioFileClip, ImageClip
from playwright.sync_api import sync_playwright

browser_type = os.getenv('AITA_BROWSER', 'chromium')
assert browser_type in ('chromium', 'firefox', 'webkit')


played_dht = os.path.join(os.path.dirname(__file__), 'played.db')
if not os.path.exists(played_dht):
  # Essentially touch the file
  diskhash.StructHash(played_dht, 7, '7s', 'rw')


def add_static_image_to_audio(image_path, audio_path, output_path):
  # https://www.thepythoncode.com/article/add-static-image-to-audio-in-python
  """Create and save a video file to `output_path` after
  combining a static image that is located in `image_path`
  with an audio file in `audio_path`"""
  # create the audio clip object
  audio_clip = AudioFileClip(audio_path)
  # create the image clip object
  image_clip = ImageClip(image_path)
  # use set_audio method from image clip to combine the audio with the image
  video_clip = image_clip.set_audio(audio_clip)
  # specify the duration of the new clip to be the duration of the audio clip
  video_clip.duration = audio_clip.duration
  # write the resuling video clip
  video_clip.write_videofile(output_path, fps=2)


def get_videos(posts):
  tb = diskhash.StructHash(played_dht, 7, '7s', 'r')
  with sync_playwright() as p:
    browser = p[browser_type].launch()
    page = browser.new_page()
    for selftext, url in posts:
      redd_id = url.split('/')[6]
      if tb.lookup(redd_id): # will be none by default
        continue
      if not os.path.exists(f'{redd_id}.mp4'):
        if not os.path.exists(f'{redd_id}.png'):
          print(f"capture {redd_id}")
          page.goto(url)
          page.mouse.wheel(0, 224)
          page.screenshot(path=f'{redd_id}.png')
        if not os.path.exists(f'{redd_id}.mp3'):
          print(f"tts {redd_id}")
          tts = gTTS(selftext, lang='en')
          redd_id = url.split('/')[6]
          with open(f'{redd_id}.mp3', 'wb') as f:
            tts.write_to_fp(f)
        add_static_image_to_audio(f'{redd_id}.png', f'{redd_id}.mp3', f'{redd_id}.mp4')
    browser.close()


if __name__ == '__main__':
  from get_posts import top_posts_text_url
  os.chdir(os.path.join(os.path.dirname(__file__), 'streams'))
  for selftext, url in top_posts_text_url('AITAFiltered'):
    get_videos([[selftext, url]])
    break
