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

# Download the dataset
1) Go back up one folder and go to the dataset folder. (`cd ../dataset`)

2) Run the following commands
```sh
python download_inference_posts.py
python clean_csvs.py
```

# Inference
1) Run `python get_audio.py` in the old terminal window. `python get_audio.py -limit 1` to only download one AITA.

2) Run `python play_audio.py localhost`. `python play_audio.py localhost -glob {redd_id}.mp3` where `redd_id` is the ID of the reddit submission plays a single clip.

# Streaming to OBS
1) Add the following configurations to your .env file.
```sh
AITA_STREAM_URL=udp://localhost:1234
AITA_STREAM_FORMAT=mpegts
```
You can change 1234 to your favorite somewhat large number and try it out to find a port that isn’t used.

2) Follow the normal steps in the "Inference" section from above. After setting up OBS, you can start playing a sequence of AITA clips and not get interrupted when `play_audio.py` runs out of clips. You can just infer more AITA responses while the old clips are playing and just run `play_audio.py` again and it will seamlessly play the new sequences to OBS, which stream them elsewhere. The `__main__.py` in this folder sequences the `get_audio` and `play_audio` scripts so that they run at the same time and could technically run forever if the dataset was continually updated with new AITA posts.

3) In OBS, create a "Media Source", uncheck "Local File", and paste the same address you set for `AITA_STREAM_URL` in the box that says "Input". Set "Reconnect Delay" to "1 S". Adjust the little red box to cover the entire screen with the Reddit screenshot.

4) Connect your Twitch account and use OBS to stream there.

5) Profit.
