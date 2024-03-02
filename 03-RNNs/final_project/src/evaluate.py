import torch

SOS_token = 0
EOS_token = 1

def evaluate(model, Q_vocab, A_vocab, pair):
    with torch.no_grad():
        # Get Question and Answer tensor
        Q_tensor = Q_vocab.get_tensor_from_sentence(pair[0])
        A_tensor = A_vocab.get_tensor_from_sentence(pair[1])

        answer_words = []

        # Get predictions
        output = model(Q_tensor, A_tensor)

        for tensor in output['decoder_output']:
            _, top_token = tensor.data.topk(1)
            if top_token.item() == 1:
                break
            else:
                word = A_vocab.index2word[top_token.item()]
                answer_words.append(word)
                
    return answer_words
    