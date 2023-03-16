import abc
import os
from typing import *

from dotenv import load_dotenv
load_dotenv()

def _import_generate_comments():
  global generate_comments, split_sort_comments, _import_generate_comments
  # change folders to import 'comment_generator' in the root
  # bad practice, but I don't want to break too many things
  cwd = os.getcwd()
  os.chdir(os.path.join(os.path.dirname(__file__), '../'))
  from comment_generator import generate_comments
  from sort_AI_comments import split_sort_comments
  os.chdir(cwd)
  _import_generate_comments = lambda: None

sorted_comments_csv = os.getenv('AITA_SORTED_AI_COMMENTS_CSV', 'sorted_AI_comments.csv')
if sorted_comments_csv:
  # Load the prefetched AI comments to be used in PolicyCommenter
  import pandas as pd
  _comments_df = pd.read_csv(
    os.path.join(os.path.dirname(__file__), '..', sorted_comments_csv),
    index_col=0
  )
else:
  _import_generate_comments()
  _comments_df = None


class Commenter(abc.ABC):
  """
  This class represents a Redditer that makes comments on posts. It takes in a
  post in the format defined by "posts.csv" in the dataset folder and gives
  a comment.
  """

  @abc.abstractmethod
  def __call__(self, post: dict) -> str: ...

class OpenAICommenter(Commenter):
  """
  A Commenter that directly asks a OpenAI large language model for a comment
  and returns that.
  """
  # def __init__(self):
  #   _import_generate_comments()
  def __call__(self):
    # TODO: use prompt_type_fixed
    return generate_comments(post['title'], post['selftext'])[0]

Policy = Callable[[dict, List[str]], int]

class PolicyCommenter(Commenter):
  """
  A Commenter that wraps an RL model. The RL model takes in the post as a
  dictionary as well as a list of candinate comments from an OpenAI large
  language model and returns an index in that list of comments for which
  comment is the best.
  """
  def __init__(self, policy: Policy):
    self.policy = policy
  def __call__(self, post):
    if _comments_df is None:
      # Use OpenAI API
      generated_comments = generate_comments(post['title'], post['selftext'])[0]
      comments = split_sort_comments(generated_comments)
      if comments[0] == '':
        return None
    else:
      # Use CSV
      redd_id = post['id']
      try:
        comments = list(_comments_df.loc[redd_id])
      except KeyError:
        return None

    return comments[self.policy(post, comments)]

class GreedyContextlessPolicy:
  """
  A policy that uses an estimate of the reward/value for a given action and
  chooses the best action.
  """
  def __init__(self, reward_model: 'sklearn.base.BaseEstimator'):
    self.reward_model = reward_model
  def __call__(self, post, comments):
    return self.reward_model.predict(comments).argmax()

def from_reward(reward_model: 'sklearn.base.BaseEstimator') -> PolicyCommenter:
  return PolicyCommenter(GreedyContextlessPolicy(reward_model))
