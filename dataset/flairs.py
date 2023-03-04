import enum

class Flair(str, enum.Enum):
  YTA = "Asshole"
  NTA = "Not the A-hole"
  ESH = "Everyone Sucks"
  NAH = "No A-holes here"
  @property
  def filter(self):
    return f'flair:"{self.value}"'
