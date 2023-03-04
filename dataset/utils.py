import enum

class Flair(str, enum.Enum):
  YTA = "Asshole"
  NTA = "Not the A-hole"
  ESH = "Everyone Sucks"
  NAH = "No A-holes here"
  @property
  def filter(self):
    return f'flair:"{self.value}"'

class PostTable:
  __slots__ = (
    'title',
    'timestamps',
    'post_ids',
    'urls',
    'self_texts',
    'score',
    'edited',
    'link_flair_texts',
    'num_comments'
  )

  def __init__(self):
    for k in self.__slots__:
      setattr(self, k, [])

  def append(self, post):
    self.title.append(post['title'])
    self.timestamps.append(int(post['created_utc']))
    self.self_texts.append(post['selftext'])
    self.post_ids.append(post['id'])
    self.urls.append(post['url'])
    self.score.append(post['score'])
    self.edited.append(post['edited'])
    self.link_flair_texts.append(post['link_flair_text'])
    self.num_comments.append(post['num_comments'])

  def pop(self):
    for k in self.__slots__:
      getattr(self, k).pop()

  def to_dict(self):
    return {
        'id':self.post_ids,
        'url':self.urls,
        'title':self.title,
        'timestamp':self.timestamps,
        'score':self.score,
        'link_flair_text':self.link_flair_texts,
        'num_comments':self.num_comments,
        'edited':self.edited,
        'selftext':self.self_texts
    }
