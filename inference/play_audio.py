#!/usr/bin/env python3
import ffmpeg
import os
import glob
import subprocess
import time
import sqlite3


inference_folder = os.path.dirname(__file__)

played_dht = os.path.join(inference_folder, 'played.db')

def stream(redd_ids, *destinations):
  # This command was converted to Python from "play_audio_pylivestream.py"
  # and was modified to concatenate all of the clips together into one long
  # stream.
  segments = []
  for seg in redd_ids:
    audio_probe = ffmpeg.probe(f'{seg}.mp3')['streams'][0]
    image = ffmpeg.input(
      f'{seg}.png',
      format='image2',
      t=float(audio_probe['duration']) + 2,
      re=None,
      loop=1
    )
    audio = ffmpeg.input(f'{seg}.mp3')
    segments.append(image)
    segments.append(audio)

  video = ffmpeg.concat(*segments, v=1, a=1)

  out = video.output(
    *destinations,
    vcodec='libx264',
    pix_fmt='uyvy422',
    preset='veryfast',
    video_bitrate='2500k',
    r=30.0,
    g=60.0,
    acodec='aac',
    audio_bitrate=128000,
    ar=44100,
    maxrate='2500k',
    bufsize='1250k',
    shortest=None,
    strict='experimental',
    format='flv'
  ).global_args('-loglevel', 'error')

  return out

if __name__ == '__main__':
  try:
      cur = sqlite3.connect(played_dht)

      os.chdir(os.path.join(inference_folder, 'streams'))

      # Get the posts that are not removed
      removed_posts = cur.execute('SELECT PostId FROM Posts WHERE Removed')
      remove_from = {post_id for post_id, in removed_posts.fetchall()}
      redd_ids = [
        redd_id
        for f in glob.iglob('*.mp3')
        if (redd_id := f.split('.', 1)[0]) not in remove_from
      ]

      # Start a RTMP server on localhost for us to stream to
      subprocess.Popen([
        'ffplay',
        '-loglevel', 'error',
        '-timeout', '5',
        '-autoexit',
        'rtmp://localhost'
      ])
      time.sleep(1)

      # Stream to localhost
      stream(redd_ids, 'rtmp://localhost/').run()

      # Increment the number of times in which the post was read out loud in
      # the database
      for redd_id in redd_ids:
        cur.execute("""
        INSERT OR IGNORE INTO Posts VALUES (?, 0, 0);
        """, (redd_id,))
        cur.execute("""
        UPDATE Posts SET Played = Played + 1 WHERE PostId LIKE ?;
        """, (redd_id,))
      cur.commit()

  finally:
      cur.close()
