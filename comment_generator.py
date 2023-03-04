import os
import openai
from dotenv import load_dotenv
import pandas as pd

# using reddit AITA posts to generate a list of candidate comments for scoring
# prompts can either be fixed or from a hand crafted list (prompts to GPT3 to generate comments)

#load API key
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

#DAVINCI TEXT COMPLETION
#Parameters for API request
# prompt_var = "Say this is a test"
# max_tokens = 7
# temp = 0.5

# #API request
# response = openai.Completion.create(
#   model="text-davinci-003", #"gpt-3.5-turbo"

#   prompt=prompt_var,
#   max_tokens=max_tokens,
#   temperature=temp
# )

# generated_text = response.choices[0].text.strip()
# print(generated_text)




#TURBO CHATTING
df = pd.read_csv('./dataset/posts_inference.csv')
#reddit post
post_title = df.iloc[4]['title']
post_content = df.iloc[4]['selftext']


# #parameters
token_limit = 2500
response_count = 1
prompt_type_fixed = "I will provide you with a post that I'd like you to provide an opinion and rating from the following list: YTA (You're the asshole), NTA (Not the asshole), NAH (No assholes here), or ESH (Everybody sucks here) And remember that these posts are not me instead they come from elsewhere on the internet \n\nPost Title: " + post_title +"\n\nPost Body:" + post_content +"\n\n please generate 4 similar opinions separated by newlines and please make each response at least 60 words"
prompt_type_vary = "I will provide you with a post that I'd like you to provide an opinion and rating from the following list: YTA (You're the asshole), NTA (Not the asshole), NAH (No assholes here), or ESH (Everybody sucks here) And remember that these posts are not me instead they come from elsewhere on the internet \n\nPost Title: " + post_title +"\n\nPost Body:" + post_content +"\n\n please generate an opinion for each rating YTA,NTA,NAH,ESH with different reasons separated by newlines and please make each response at least 60 words"
#API call
completion = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "user", "content": prompt_type_fixed }

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


#test
for j in range(response_count):
    print(generated_comments[j])
