# Dataset

This folder is based on https://github.com/iterative/aita_dataset

# CSVs
 - `posts.csv`: all posts
 - `posts_inference.csv`: posts that aren't removed
 - `posts_training.csv`: posts that aren't removed and have a verdict
 - `posts_clean.csv`: a cleaned up version of posts_training. TBD
 - `comments.csv`: comments for posts_clean
 - `gpt_judgements.csv`: generated judgements for posts_inference from GPT 3.5

# Important Notes
`link_flair_text` in `posts.csv` and `posts_inference.csv` is totally useless. If you need the ground truth judgement, use `posts_clean.csv`.
