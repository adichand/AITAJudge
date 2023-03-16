#!/usr/bin/env python3

from dotenv import load_dotenv
load_dotenv()

import itertools
import os

import praw
import pandas as pd
import tqdm

dataset_folder = os.path.dirname(os.path.abspath(__file__))
os.chdir(dataset_folder)

from utils import Flair, PostTable, CommmentTable

os.chdir('..')
reddit = praw.Reddit()
os.chdir(dataset_folder)

post_table = PostTable()

limit = 20
count = limit * len(Flair)

comment_table = CommmentTable()

subreddit = reddit.subreddit('AmItheAsshole')
for post_obj in tqdm.tqdm(itertools.chain.from_iterable(
    # subreddit.search(Flair.ESH.filter, sort='new', limit=1000)
    subreddit.search(flair.filter, sort='top', limit=limit)
    for flair in Flair
), total=count):
    post_table.append(vars(post_obj))
    comment_table.append(post_obj)
    # TODO: do something with comments

# Write to a csv file
df = pd.DataFrame(post_table.to_dict())
output_path = "posts_training.csv"
df.to_csv(output_path, index=False, mode='a', header=not os.path.exists(output_path))

df = pd.DataFrame(comment_table.to_dict())
output_path = "comments.csv"
df.to_csv(output_path, index=False, mode='a', header=not os.path.exists(output_path))
