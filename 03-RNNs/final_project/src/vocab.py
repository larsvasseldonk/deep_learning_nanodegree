import torch

SOS_token = 0
EOS_token = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Vocab:
    def __init__(self):
        self.word2index = {} 
        self.word2count = {}
        self.index2word = {SOS_token: "SOS", EOS_token: "EOS"}
        self.n_words = 2  # Count SOS and EOS
       
    # Split a sentence into words and add it to the vocab  
    def add_sentence(self, sentence):
        for word in sentence.split(" "):
            self.add_word(word)

    # Add new words to the vocab, if the word already exists,
    # the word counter will be updated
    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def get_indexes_from_sentence(self, sentence):
        indexes = []
        for word in sentence.split(" "):
            try:
                indexes.append(self.word2index[word])
            except:
                raise Exception(f"Error: {word} not in the vocabulary")
        return [self.word2index[word] for word in sentence.split(" ")]

    def get_tensor_from_sentence(self, sentence):
        indexes = self.get_indexes_from_sentence(sentence)
        # Add End Of Sentence (EOS) token
        indexes.append(EOS_token)
        return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def get_tensors_from_pair(Q_vocab: Vocab, A_vocab: Vocab, pair: tuple[str, str]):
    Q_tensor = Q_vocab.get_tensor_from_sentence(pair[0])
    A_tensor = A_vocab.get_tensor_from_sentence(pair[1])
    return (Q_tensor, A_tensor)
