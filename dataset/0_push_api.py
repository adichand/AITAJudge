# We go through the push API to get the ids of every post, then use the official reddit
# API to get the contents of each post and metadata of interest.
import requests
import json
import pandas as pd
import time

first_epoch = '24h'
last_epoch = int(time.time())

def getPushshiftData(after, before):
    #sortby = 'created_utc'
    #order = 'asc'
    sortby = 'score'
    order = 'desc'
    url = f'https://api.pushshift.io/reddit/submission/search/?sort={sortby}&order={order}&subreddit=amitheasshole&after={str(after)}&before={str(before)}&size=1000'
    print(url)
    r = requests.get(url)
    data = json.loads(r.text)
    print(data)
    return data['data']

title = list()
timestamps = list()
post_ids = list()
urls = list()
self_texts = list()
score = list()
edited = list()
link_flair_texts = list()
num_comments = list()

after = first_epoch
first = True
while first or after < last_epoch:
    data = getPushshiftData(after, last_epoch)
    for post in data:
        title.append(post['title'])
        timestamps.append(int(post['created_utc']))
        self_texts.append(post['selftext'])
        post_ids.append(post['id'])
        urls.append(post['url'])
        score.append(post['score'])
        # print(post['score'])
        edited.append(post['edited'])
        link_flair_texts.append(post['link_flair_text'])
        num_comments.append(post['num_comments'])
    if len(post_ids) > 1 and post_ids[-2] == post_ids[-1]:
        title.pop()
        timestamps.pop()
        self_texts.pop()
        post_ids.pop()
        urls.pop()
        score.pop()
        edited.pop()
        link_flair_texts.pop()
        num_comments.pop()
        break
    after = timestamps[-1]
    print([str(len(post_ids)) + " posts collected so far."])
    time.sleep(0.1)
    first = False

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
df.to_csv("posts.csv", index=False)
