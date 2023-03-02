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

2) Run `python get_audio.py`

3) Run `python -m pylivestream.loopfile streams/{redd_id}.mp4 localhost pylivestream.json` where `redd_id` is the ID of the reddit submission
