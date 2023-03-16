#!/usr/bin/env python3
import ffmpeg
from contextlib import closing
import os
import glob
import subprocess
import time
import sqlite3
import shlex
import stat


from dotenv import load_dotenv
load_dotenv()


inference_folder = os.path.dirname(__file__)

played_dht = os.path.join(inference_folder, 'played.db')

stream_url = os.getenv('AITA_STREAM_URL', None)
stream_urls = ('rtmp://localhost',) if stream_url is None else shlex.split(stream_url)
stream_format = os.getenv('AITA_STREAM_FORMAT', 'flv')

def stream(redd_ids, *destinations):
  # This command was converted to Python from "play_audio_pylivestream.py"
  # and was modified to concatenate all of the clips together into one long
  # stream.
  # https://github.com/adichand/AITAJudge/blob/6b8f6b/inference/play_audio_pylivestream.py
  segments = []
  for seg in redd_ids:
    duration = ffmpeg.probe(f'{seg}.mp3')['streams'][0]['duration']

    image = ffmpeg.input(
      f'{seg}.png',
      format='image2',
      t=float(duration) + 2,
      re=None,
      loop=1
    )
    # The main text
    audio = ffmpeg.input(f'{seg}.mp3')
    segments.append(image)
    segments.append(audio)

  # # Silence
  # audio = ffmpeg.input(
  #   'anullsrc=channel_layout=mono:sample_rate=24000',
  #   format='lavfi',
  #   t=1 # some padding after the stream so that FFMPEG doesn't close early
  # )
  # segments.append(image)
  # segments.append(audio)

  video = ffmpeg.concat(*segments, v=1, a=1)

  out = video.output(
    *destinations,
    vcodec='libx264',
    pix_fmt='uyvy422',
    preset='veryfast',
    video_bitrate='2500k',
    r=30.0, # 3.0
    g=60.0,
    acodec='aac',
    audio_bitrate=128000,
    ar=44100,
    maxrate='2500k',
    bufsize='1250k',
    shortest=None,
    strict='experimental',
    format=stream_format
  ).global_args('-loglevel', 'error')

  return out

def stream_valid(condition='Removed', is_async=False):
  with closing(sqlite3.connect(played_dht)) as con:
    # Get the posts that are not removed
    with closing(con.cursor()) as cur:
      cur.execute('SELECT PostId FROM PostsPlayed WHERE ' + condition)
      remove_from = {post_id for post_id, in cur.fetchall()}
    redd_ids = [
      redd_id
      for f in glob.iglob('*.mp3')
      if (redd_id := f.split('.', 1)[0]) not in remove_from
      if not f.startswith('tmp_')
    ]

    # Don't do anything if there are no clips to play.
    if len(redd_ids) == 0:
      return

    if stream_url is None:
      # Start a RTMP server on localhost for us to stream to
      subprocess.Popen([
        'ffplay',
        '-loglevel', 'error',
        '-timeout', '5',
        '-autoexit',
        'rtmp://localhost'
      ])
      time.sleep(1)

    # Stream to the URLs specified
    s = stream(redd_ids, *stream_urls)
    if is_async:
      s.run_async()
    else:
      s.run()

    # Increment the number of times in which the post was read out loud in
    # the database
    redd_ids2 = [(redd_id,) for redd_id in redd_ids]
    with closing(con.cursor()) as cur:
      cur.executemany("""
      INSERT OR IGNORE INTO PostsPlayed VALUES (?, 0, 0);
      """, redd_ids2)
      cur.executemany("""
      UPDATE PostsPlayed SET Played = Played + 1 WHERE PostId LIKE ?;
      """, redd_ids2)
    con.commit()

    # Delete the temporary files if they are named pipes
    for redd_id in redd_ids:
      for path in [
        f'{redd_id}.mp3',
        f'{redd_id}.png',
      ]:
        if stat.S_ISFIFO(os.stat(path).st_mode):
          os.remove(path)

if __name__ == '__main__':
  os.chdir(os.path.join(inference_folder, 'streams'))
  stream_valid()
