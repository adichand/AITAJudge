import os
import pickle

from models.sklearn_models.embed_sentences import get_model, get_sentence_embedding
from models.sklearn_models.embed_sentences import MODEL_FILENAME

SAMPLE_COMMENTS = ["NTA - You're not the asshole for hurting everybody.", "YTA - Dude, you shouldn't be hurting people?"]

SAVED_MODEL_PATH = "./saved_models/nn_regressor_nrows=10000_generic.pkl"


def main():
    # Getting vector model
    w2v_model = get_model(MODEL_FILENAME)

    model = None
    with open(SAVED_MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    print(type(model))

    comment_embeds = [get_sentence_embedding(w2v_model, comment) for comment in SAMPLE_COMMENTS]

    preds = model.predict(comment_embeds)

    print(preds)


if __name__ == "__main__":
    main()
