# Deprecated. Use get_audio.py
import textwrap

from dotenv import load_dotenv
load_dotenv()

from playwright.sync_api import sync_playwright
import praw

import os

browser_type = os.getenv('AITA_BROWSER', 'chromium')
assert browser_type in ('chromium', 'firefox', 'webkit')

reddit = praw.Reddit()
subreddit = reddit.subreddit('AmItheAsshole')
# to loop: for submission in subreddit.stream.submissions():
submission = next(subreddit.stream.submissions())
selftext = submission.selftext
url = submission.url

print('\n'.join(textwrap.wrap(selftext)))

with sync_playwright() as p:
    browser = p[browser_type].launch()
    page = browser.new_page()
    page.goto(url)
    page.mouse.wheel(0, 224)
    page.screenshot(path=f'screenshot.png')
    browser.close()
