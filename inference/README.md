# Setup

1) Install ffmpeg from conda if it isn't already on your machine (Linux and Mac probably already have it)

2) Run the following commands
```sh
pip install -r ../requirements.txt
playwright install chromium
```

3) (Optional) If you care, mount tmpfs on the streams folder so that the individual videos to stream live in RAM.
This varies from OS to OS. On Linux, it might be `sudo mount -t tmpfs -o size=500m tmpfs streams`. On Mac, it might be `sudo mount_tmpfs streams`.

# Inference
1) In a new terminal window, run `python -m pyrtmp.rtmp` to start up a RTMP server on your computer

2) Run `python get_audio.py` in the old terminal window. `python get_audio.py -limit 1` to only download one AITA.

3) Run `python play_audio.py localhost`. `python play_audio.py localhost -glob {redd_id}.mp3` where `redd_id` is the ID of the reddit submission plays a single clip.
