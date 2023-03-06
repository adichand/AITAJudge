# Setup

## Option 1: conda
1) Install Chrome if it isn't already on your machine

2) Run the following commands
```sh
conda env create -f ../environment.yml
conda activate aita-judge
```

## Option 2: pip
1) Install Chrome if it isn't already on your machine

2) Install ffmpeg if it isn't already on your machine (Linux and Mac probably already have it)

3) Run the following command
```sh
pip install -r ../requirements.txt
```

# Inference
1) Run `python get_audio.py` in the old terminal window. `python get_audio.py -limit 1` to only download one AITA.

2) Run `python play_audio.py localhost`. `python play_audio.py localhost -glob {redd_id}.mp3` where `redd_id` is the ID of the reddit submission plays a single clip.
