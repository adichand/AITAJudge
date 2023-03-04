#!/usr/bin/env python3

from dotenv import load_dotenv
load_dotenv()

import itertools
import os

import praw
import pandas as pd
import tqdm

dataset_folder = os.path.dirname(__file__)
os.chdir(dataset_folder)

from flairs import Flair

os.chdir('..')

title = list()
timestamps = list()
post_ids = list()
urls = list()
self_texts = list()
score = list()
edited = list()
link_flair_texts = list()
num_comments = list()

limit = 1000
count = limit * len(Flair)

reddit = praw.Reddit()
subreddit = reddit.subreddit('AmItheAsshole')
for post_obj in tqdm.tqdm(itertools.chain.from_iterable(
    # subreddit.search(Flair.ESH.filter, sort='new', limit=1000)
    subreddit.search(flair.filter, sort='new', limit=limit)
    for flair in Flair
), total=count):
    post = vars(post_obj)
    title.append(post['title'])
    timestamps.append(int(post['created_utc']))
    self_texts.append(post['selftext'])
    post_ids.append(post['id'])
    urls.append(post['url'])
    score.append(post['score'])
    edited.append(post['edited'])
    link_flair_texts.append(post['link_flair_text'])
    num_comments.append(post['num_comments'])

    comments = post_obj.comments
    # TODO: do something with comments

os.chdir(dataset_folder)

# Write to a csv file
d = {
    'id':post_ids,
    'url':urls,
    'title':title,
    'timestamp':timestamps,
    'score':score,
    'link_flair_text':link_flair_texts,
    'num_comments':num_comments,
    'edited':edited,
    'selftext':self_texts
}
df = pd.DataFrame(d)
output_path = "posts_training.csv"
df.to_csv(output_path, index=False, mode='a', header=not os.path.exists(output_path))
