import abc
import os
from typing import *

class Commenter(abc.ABC):
  @abc.abstractmethod
  def __call__(self, post: dict) -> str: ...

class PolicyCommenter(Commenter):
  def __init__(self, policy: Callable[int, Tuple[dict, List[str]]]):
    import pandas as pd
    self.comments_df = pd.read_csv(
      os.path.join(os.path.dirname(__file__), '../sorted_AI_comments.csv'),
      index_col=0
    )
    self.policy = policy
  def __call__(self, post):
    redd_id = post['id']
    try:
      comments = list(self.comments_df.loc[redd_id])
    except KeyError:
      print(f'no AI comment {redd_id}')
      return None

    return comments[self.policy(post, comments)]
