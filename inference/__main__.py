#!/usr/bin/env python3
import asyncio
import threading
import csv
import os
import sys
import queue

inference_folder = os.path.dirname(__file__)
os.chdir(inference_folder)

from play_audio import stream_valid
from get_audio import get_videos

os.chdir(os.path.join(inference_folder, 'streams'))

dataset_path = '../dataset/posts_inference.csv'
limit = 10

new_snips = 1


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
  new_snips = 1
  while new_snips != 0:
    # Hopefully, it takes much longer to play the audio than it does to generate it.
    t = ExThread(target=stream_valid, args=('Removed OR Played > 0',))
    t.start()

    with open(os.path.join('..', dataset_path), 'r') as f:
      new_snips = await get_videos(csv.DictReader(f), limit)

    t.join_with_exception()


asyncio.run(main())
