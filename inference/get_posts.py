# Deprecated. Use dataset/0_push_api.py
import requests

import datetime
import textwrap
import shelve
import typing
import os

shelf_loc = os.path.join(os.path.dirname(__file__), 'reddit-cache')


class Post(typing.NamedTuple):
  selftext: str
  url: str

_subreddits_content_getters = {}
def register(sub):
  def decorator(func):
    _subreddits_content_getters[sub] = func
    return func
  return decorator

@register('AITAFiltered')
def _(post):
  ipost = post['data']['crosspost_parent_list'][0]
  selftext = ipost['selftext']
  url = ipost['url']
  return Post(selftext, url)


def get_top_json(sub):
  with shelve.open(shelf_loc) as d:
    if sub in d:
      top_json, fetch_time = d[sub]
    else:
      fetch_time = datetime.datetime.now()
      top_json = requests.get(f'https://reddit.com/r/{sub}/top.json?show=all&limit=100').json()
      if top_json.get('error') == 429:
        print('Wait')
        raise
      d[sub] = top_json, fetch_time
  return top_json, fetch_time

def flush_json(sub):
  with shelve.open(shelf_loc) as d:
    del d[sub]

def top_posts(sub):
  top_json, fetch_time = get_top_json(sub)

  # import pprint
  # pprint.pprint(top_json)
  # print(fetch_time)

  posts = top_json['data']['children']
  return posts

def top_posts_text_url(sub):
  for post in top_posts(sub):
    selftext, url = _subreddits_content_getters[sub](post)
    yield selftext, url

def show_posts(sub):
  for post in top_posts(sub):
    selftext, url = _subreddits_content_getters[sub](post)
    print(url)
    if selftext == '[deleted]':
      continue
    print('\n'.join(textwrap.wrap(selftext)))
    print()

if __name__ == '__main__':
  show_posts('AITAFiltered')


# https://reddit.com/r/redditdev/comments/t9dems/praw_reddit_api_get_top_comment_for_each_post/
