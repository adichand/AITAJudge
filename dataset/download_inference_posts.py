#!/usr/bin/env python3
# We go through the push API to get the ids of every post, then use the official reddit
# API to get the contents of each post and metadata of interest.
import requests
import json
import pandas as pd
import time
import os

dataset_folder = os.path.dirname(__file__)
os.chdir(dataset_folder)

from utils import PostTable

first_epoch = '7d'
last_epoch = '6d'

# now = time.time()
now = pd.to_datetime('2023-03-10T13:00:00-08:00').timestamp() # reproducability

if first_epoch is None:
    first_epoch = int(now)
elif isinstance(first_epoch, str):
    first_epoch = int(now - pd.Timedelta(first_epoch).total_seconds())

last_epoch_str = last_epoch
if last_epoch is None:
    last_epoch = int(now)
elif isinstance(last_epoch, str):
    last_epoch = int(now - pd.Timedelta(last_epoch).total_seconds())

def getPushshiftData(after, before, sortby='created_utc', order = 'asc'):
    url = f'https://api.pushshift.io/reddit/submission/search/?sort={sortby}&order={order}&subreddit=amitheasshole&after={str(after)}&before={str(before)}&size=1000'
    print(url)
    r = requests.get(url)
    data = json.loads(r.text)
    print(data)
    return data['data']

post_table = PostTable()
post_ids = post_table.post_ids
timestamps = post_table.timestamps

assert first_epoch < last_epoch

after = first_epoch
while after < last_epoch:
    data = getPushshiftData(after, last_epoch)
    for post in data:
        post_table.append(post)
    if len(post_ids) > 1 and post_ids[-2] == post_ids[-1]:
        post_table.pop()
        break
    after = timestamps[-1]
    print([str(len(post_ids)) + " posts collected so far."])
    time.sleep(0.1)

# Write to a csv file
df = pd.DataFrame(post_table.to_dict())
df.to_csv("posts.csv", index=False, mode='a', header=not os.path.exists("posts.csv"))
