import os
import openai
from dotenv import load_dotenv

# using reddit AITA posts to generate a list of candidate comments for scoring
# prompts can either be fixed or from a hand crafted list (prompts to GPT3 to generate comments)

#load API key
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

#Parameters for API request
prompt_var = "Say this is a test"
max_tokens = 7
temp = 0.5

#API request
response = openai.Completion.create(
  model="text-davinci-003",
  prompt=prompt_var,
  max_tokens=max_tokens,
  temperature=temp
)

generated_text = response.choices[0].text.strip()

print(generated_text)