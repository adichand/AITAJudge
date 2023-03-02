#!/usr/bin/env python3
# copied from https://github.com/scivision/PyLivestream/blob/main/src/pylivestream/glob.py
from __future__ import annotations
import random
from pathlib import Path
import signal
import argparse
import os

from pylivestream.base import FileIn
from pylivestream.glob import fileglob
from pylivestream.ffmpeg import get_meta


def stream_files(
    ini_file: Path,
    websites: list[str],
    *,
    video_path: Path,
    glob: str = None,
    assume_yes: bool = False,
    loop: bool = None,
    shuffle: bool = None,
    image_suffix: str = None,
):
    # %% file / glob wranging
    flist = fileglob(video_path, glob)

    print("streaming these files. Be sure list is correct! \n")
    print("\n".join(map(str, flist)))
    print()

    if assume_yes:
        print("going live on", websites)
    else:
        input(f"Press Enter to go live on {websites}.    Or Ctrl C to abort.")

    if loop:
        while True:
            playonce(flist, image_suffix, websites, ini_file, shuffle, assume_yes)
    else:
        playonce(flist, image_suffix, websites, ini_file, shuffle, assume_yes)


def playonce(
    flist: list[Path],
    image_suffix: str,
    sites: list[str],
    inifn: Path,
    shuffle: bool,
    yes: bool,
):

    if shuffle:
        random.shuffle(flist)

    image: Path

    if image_suffix == '':
        image_suffix = None

    for f in flist:
        # TODO: pad audio files so that ffmpeg doesn't cut off that last two seconds?
        # meta: dict = get_meta(f)
        # timeout: float = float(meta['streams'][0]['duration']) + 2

        if image_suffix:
            image = f.with_suffix(image_suffix)
        else:
            image = None

        s = FileIn(
            inifn, sites, infn=f, loop=False, image=image, caption=None, yes=yes, timeout=None
        )

        s.golive()


def cli():
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    p = argparse.ArgumentParser(description="Livestream a globbed input file list")
    p.add_argument(
        "websites",
        help="site to stream, e.g. localhost youtube facebook twitch",
        nargs="+",
    )
    p.add_argument("-path", help="path to discover files from", default='')
    p.add_argument("-json", help="JSON file with stream parameters such as key", default='../pylivestream.json')
    p.add_argument("-glob", help="file glob pattern to stream.", default='*.mp3')
    p.add_argument("-image-suffix", help="suffix of static image to display, for audio-only files.", default='.png')
    p.add_argument("-shuffle", help="shuffle the globbed file list", action="store_true")
    p.add_argument("-loop", help="repeat the globbed file list endlessly", action="store_true")
    p.add_argument("-y", "--yes", help="no confirmation dialog", action="store_true")
    P = p.parse_args()

    stream_files(
        ini_file=P.json,
        websites=P.websites,
        assume_yes=P.yes,
        loop=P.loop,
        video_path=P.path,
        glob=P.glob,
        shuffle=P.shuffle,
        image_suffix=P.image_suffix,
    )


if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(__file__), 'streams'))
    cli()
