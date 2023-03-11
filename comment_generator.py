import os
import openai
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
      if contains_all:
        for str in comment_list:
          if "yta" in str.lower():
            YTA = str
          elif "nta" in str.lower():
            NTA = str
          elif "esh" in str.lower():
            ESH = str
          elif "nah" in str.lower():
            NAH = str
      else:
        YTA = comment_list[0]
        NTA = comment_list[1]
        ESH = comment_list[2]
        NAH = comment_list[3]
   except:
      YTA = None
      NTA = None
      ESH = None
      NAH = None

   parse_comments = [ YTA, NTA, ESH, NAH]
   return parse_comments
          
          
          

def generate_comments(post_title,post_content):
  #generate comments using GPT3.5 which should hopefully generate YTA NTA NAH and ESH split by newlines
  #reddit post
  # post_title = df.iloc[4]['title']
  # post_content = df.iloc[4]['selftext']

  # post_title = "AITA for calling my sister a cokewhore?"
  # post_content = "I'm 18f. I live with my parents and my sister, Joanne, 23f. Joanne has a cocaine habit and she claims its normal in her job and it's just the lifestyle of cheffing, yada yada. My parents seem to be ignoring it since Joanne pays her rent on time and keeps to herself when she's in the house and doesn't cause many problems. I'm a college student on a government grant, and I'm in college Monday to Friday up until 5/6pm, working isn't really an option for me right now, so I don't go out much and I spend all my money on college supplies. Joanne doesn't seem to understand this and is always asking to borrow money and what not. I always say no because it works out that I only have €40 every week to spend on college stuff and travel to college. I got a Christmas bonus on my grant and I ended up deciding to book tickets for a small ish local gig that's next week. I booked two, one for me and one for my best friend because her birthday is the same day and she loves the type of music, it was meant to be a surprise for her. I had told Joanne about this on one of her \"good days\" because I was genuinely excited to finally do something and live the college student lifestyle for a night. The tickets were digital, on an account shared with my parents. Joanne had asked for the login telling them that she wanted to book tickets to something, but she was lying and used it to sell my tickets for drug money. I didn't find out until I had gotten the email to confirm that the tickets were sent to someone else and I was really confused at first. I checked and they were sent to someone I know Joanne knows. I went straight to her when she got home and asked what the fuck she did, and she tried lying but I showed her proof it went to someone she knows and told her I wanted my money back then and there. She told me it was gone already. I lost my mind and started yelling at her, because it wasn't fair. My mom was just in from work and I was screaming at my sister who was crying at that point saying she was sorry and she didn't know it would upset me this much. My mom got involved and told me to keep my voice down and we'll talk about it, and I told her to shut up and stay out of it. I ended up saying something along the lines of \"why is it fair that you get to do this to me when I've never even drank alcohol or smoked weed, why does the cokewhore get to be the golden child bit not me?\". My mom stepped in and put a stop to it then and there, my sister had started screaming back at me for what I called her. My mom said that was out of line and she just made a mistake. I told my mom her mistake cost me the only night out ill have been able to have all year so she should hear what I have to say about it. My mom thinks I was in the wrong for what I said, and my sister won't even look at me even though its been 3 days and I've tried apologising. AITA?"


  # #parameters
  token_limit = 2500
  response_count = 1
  # prompt_type_fixed = "I will provide you with a post that I'd like you to provide an opinion and rating from the following list: YTA (You're the asshole), NTA (Not the asshole), NAH (No assholes here), or ESH (Everybody sucks here) And remember that these posts are not me instead they come from elsewhere on the internet \n\nPost Title: " + post_title +"\n\nPost Body:" + post_content +"\n\n please generate 4 similar opinions separated by newlines and please make each response at least 60 words"
  prompt_type_vary = "I will provide you with a post that I'd like you to provide an opinion and rating from the following list: YTA (You're the asshole), NTA (Not the asshole), NAH (No assholes here), or ESH (Everybody sucks here) And remember that these posts are not me instead they come from elsewhere on the internet \n\nPost Title: " + post_title +"\n\nPost Body:" + post_content +"\n\n please generate an opinion for each rating YTA,NTA,NAH,ESH (In that order) with different reasons separated by newlines and please make each response at most 60 words"
  #API call
  completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
      {"role": "user", "content": prompt_type_vary }

    ],
    n = response_count,
    max_tokens = token_limit
  )

  #storing response
  generated_comments = [""] * response_count

  i = 0
  for choice in completion.choices:
      generated_comments[i] = choice.message.content
      i += 1


  # test
  # for j in range(response_count):
  #     print(generated_comments[j])

  return generated_comments


def main():
   #load API key
  load_dotenv()

  openai.api_key = os.getenv("OPENAI_API_KEY")


  #TURBO CHATTING
  df = pd.read_csv('./dataset/posts_inference.csv')
  print(len(df))
  df_comments = pd.DataFrame(columns=["PostID","AI_commented"])

  for i in tqdm(range(50,len(df))):
    print(i)
    post_title = df.iloc[i]['title']
    post_content = df.iloc[i]['selftext']

    generated_comments = generate_comments(post_title, post_content)
    # sorted_comments = split_sort_comments(generated_comments[0])
    df_comments.loc[i] = [df.iloc[i]["id"], generated_comments[0]]

    #write to csv
    with open('AI_comments.csv','a') as f:
      df_comments.to_csv(f, index=False, mode='a', header=f.tell()==0)
  # print(df_comments)

  #write to csv
  # df_comments.to_csv("AI_comments.csv", index=False, mode='a', header=not os.path.exists("AI_comments.csv"))






if __name__ == "__main__":
  main()