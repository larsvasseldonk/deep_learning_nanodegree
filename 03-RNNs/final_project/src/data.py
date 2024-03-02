import string
import nltk
from torchtext.datasets import SQuAD1
import pandas as pd

stemmer = nltk.stem.snowball.SnowballStemmer("english")
tokenizer = nltk.tokenize.RegexpTokenizer(r"\w+")

def get_data_dict(data_pipe):

    data_dict = {
        "Question": [],
        "Answer": []
    }
    
    for _, question, answers, _ in data_pipe:
        data_dict["Question"].append(stem_text(clean_text(question)))
        data_dict["Answer"].append(stem_text(clean_text(answers[0])))
        
    return data_dict


def clean_text(text):
   """Clean text by lowering it and removing special characters."""
   
   return ''.join([s.lower() for s in text if s not in string.punctuation])


def stem_text(text):
   """Stemming refers to a crude heuristic process that chops off the ends of words."""

   return ' '.join(stemmer.stem(w) for w in text.split())


def tokenize_text(text):
    """Tokenization breaks text into smaller parts for easier machine analysis."""

    return tokenizer.tokenize(text)


def get_pairs(df: pd.DataFrame):
    """Convert dataframe to list of question answer pairs."""

    questions = df["Question"].to_list()
    answers = df["Answer"].to_list()
    return [list(q_a) for q_a in zip(questions, answers)]


def load_data() -> pd.DataFrame:

    train_data, val_data = SQuAD1(split=("train", "dev"))
    train_dict, val_dict = get_data_dict(train_data), get_data_dict(val_data)

    train_df = pd.DataFrame(train_dict)
    validation_df = pd.DataFrame(val_dict)

    return pd.concat([train_df, validation_df], axis=0)