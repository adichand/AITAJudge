import os
from dotenv import load_dotenv
import pandas as pd
from tqdm import tqdm

# using reddit AITA posts to generate a list of candidate comments for scoring
# prompts can either be fixed or from a hand crafted list (prompts to GPT3 to generate comments)


#split comments into sub comments and try to sort them if possible
def split_sort_comments(comments):
   try:
      contains_all = all(substr in comments for substr in ["YTA","NTA","ESH","NAH"])
      comment_list = comments.split("\n")
      comment_list = [substr for substr in comment_list if substr != '']
      if contains_all:
        for str in comment_list:
          if "YTA" in str:
            YTA = str
          elif "NTA" in str:
            NTA = str
          elif "ESH" in str:
            ESH = str
          elif "NAH" in str:
            NAH = str
      else:
        YTA = comment_list[0]
        NTA = comment_list[1]
        ESH = comment_list[2]
        NAH = comment_list[3]
   except:
      YTA = ""
      NTA = ""
      ESH = ""
      NAH = ""

   parse_comments = [ YTA, NTA, ESH, NAH]
   return parse_comments

def main():
  df = pd.read_csv('./AI_comments.csv')
  df = df.drop_duplicates().reset_index(drop=True)
  df_sorted_comments = pd.DataFrame(columns=["PostID","YTA","NTA","ESH","NAH"])
  print(df.shape[0])
  for i in tqdm(range(df.shape[0])):
    sorted_comments = split_sort_comments(df.loc[i]["AI_commented"])
    df_sorted_comments.loc[i] = [df.iloc[i]["PostID"], sorted_comments[0], sorted_comments[1], sorted_comments[2], sorted_comments[3]]


  df_sorted_comments.drop_duplicates().reset_index(drop=True)
  with open('sorted_AI_comments.csv','a') as f:
    df_sorted_comments.to_csv(f, index=False, mode='a', header=f.tell()==0)


if __name__ == "__main__":
  main()