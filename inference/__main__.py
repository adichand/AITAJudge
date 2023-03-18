#!/usr/bin/env python3
import asyncio
import threading
import csv
import os
import sys
import queue

cwd = os.getcwd()
inference_folder = os.path.dirname(os.path.abspath(__file__))

os.chdir(inference_folder)

from play_audio import stream_valid
from get_audio import get_videos, load_model, load_posts

os.chdir(cwd)

# https://stackoverflow.com/a/6874161
class ExThread(threading.Thread):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__status_queue = queue.Queue()

    def run_with_exception(self):
        """This method should be overriden."""
        super().run()

    def run(self):
        """This method should NOT be overriden."""
        try:
            self.run_with_exception()
        except BaseException:
            self.__status_queue.put(sys.exc_info())
        self.__status_queue.put(None)

    def wait_for_exc_info(self):
        return self.__status_queue.get()

    def join_with_exception(self):
        ex_info = self.wait_for_exc_info()
        if ex_info is None:
            return
        else:
            raise ex_info[1]


async def main():
  import argparse
  p = argparse.ArgumentParser(description="Download audio snippets from gTTS")
  p.add_argument("-path", help="path of the posts.csv from the dataset", default=None)
  p.add_argument("-model", help="path of the model.pkl file", default=os.path.join(
    inference_folder,
    '../models/sklearn_models/saved_models/nn_regressor_nrows=20000_generic.pkl'
  ))
  p.add_argument("-limit", help="the number of audio clips to download per FFMPEG session", type=int, default=10)
  P = p.parse_args()

  dataset_path = P.path
  model_path = P.model
  limit = P.limit

  model = load_model(model_path)
  os.chdir(os.path.join(inference_folder, 'streams'))

  new_snips = 1
  while new_snips != 0:
    # Hopefully, it takes much longer to play the audio than it does to generate it.
    t = ExThread(target=stream_valid, args=('Removed OR Played > 0',))
    t.start()

    new_snips = await get_videos(load_posts(dataset_path), model, limit)

    t.join_with_exception()


asyncio.run(main())
