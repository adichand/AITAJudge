#!/usr/bin/env python3
# Do some cleaning to get the data in good order for a classification problem

import pandas as pd
import os

def clean_scrape(df_use):
    # Remove any edits that may give away the answer [ie, "edit: okay you're right I'm the asshole" ]
    df_use['body'] = df_use.pop('selftext').str.replace("(edit|update).*?(YTA|a-|ass|\\sta\\s)(.*)","",case=False)
    # Remove any deleted or removed posts
    gone_list = ["[deleted]","[removed]",""]
    df_use = df_use[df_use['body'].isin(gone_list)==False]
    print("After removing deleted posts, there are " +  str(len(df_use)) + " posts left.")
    return(df_use)


def merge_scrape(old, new):
    # old and new are pandas dataframes. should have the same columns
    old = pd.concat([old,new])
    old = old.drop_duplicates()
    return(old)

# Add the new cleaned results
raw = pd.read_csv("posts.csv")
grand = clean_scrape(raw)

print("There are now " +  str(len(grand)) + " cleaned posts.")

grand.to_csv("posts_inference.csv",index=False)
