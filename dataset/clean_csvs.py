#!/usr/bin/env python3
# Do some cleaning to get the data in good order for a classification problem

import pandas as pd
import sqlite3
import os

dataset_folder = os.path.dirname(os.path.abspath(__file__))
os.chdir(dataset_folder)

def clean_scrape(df_use, body='selftext'):
    # Remove duplicate entries that may have been recieved due to multiple downloads
    df_use = df_use.drop_duplicates('id')
    # Remove any deleted or removed posts
    gone_list = ["[deleted]","[removed]",""]
    df_use = df_use[df_use[body].isin(gone_list)==False]
    print("After removing deleted posts, there are " +  str(len(df_use)) + " posts left.")
    return(df_use)

# Add the new cleaned results
if os.path.exists("posts.csv"):
    raw = pd.read_csv("posts.csv", on_bad_lines='skip')
    grand = clean_scrape(raw)

    print("There are now " +  str(len(grand)) + " cleaned inference posts.")

    grand.to_csv("posts_inference.csv",index=False)

    print("Saving 'posts_inference.csv' into ../inference/played.db")

    cur = sqlite3.connect('../inference/played.db')

    grand = grand.set_index('id')
    grand.to_sql('Posts', con=cur, if_exists='replace', index_label='id', dtype={'id': 'CHAR(8) NOT NULL PRIMARY KEY'})

    # PostsPlayed table needed for get_data.py
    cur.execute("""
    CREATE TABLE IF NOT EXISTS PostsPlayed
    (
      PostId CHAR(8) NOT NULL,
      Removed BOOL,
      Played INT,
      PRIMARY KEY(PostId)
    )
    """)

    cur.commit()
    cur.close()

else:
    print("No inference posts available. Run `python download_inference_posts.py` if you want them.")

if os.path.exists("posts_training.csv"):
    raw = pd.read_csv("posts_training.csv", on_bad_lines='skip')
    grand = clean_scrape(raw)

    print("There are now " +  str(len(grand)) + " cleaned training posts.")

    grand.to_csv("posts_clean.csv",index=False)
else:
    print("No training posts available. Run `python download_training_posts.py` if you want them.")

if os.path.exists("comments.csv"):
    raw = pd.read_csv("comments.csv", on_bad_lines='skip')
    grand = clean_scrape(raw, body='body')

    print("There are now " +  str(len(grand)) + " cleaned comments.")

    grand.to_csv("comments_clean.csv",index=False)
else:
    print("No training posts available. Run `python download_training_posts.py` if you want them.")
