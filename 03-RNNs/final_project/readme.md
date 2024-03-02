# LSTM Chatbot

## Project Instructions
### Submission Instructions
You'll be using a workspace within the Udacity classroom to submit your project.

Once you submit your code, a reviewer will access your project and grade it according to the Udacity grading rubric. You can access the rubric here.

### Instructions Summary
The LSTM Chatbot will help you show off your skills as a deep learning practitioner. You will develop the chatbot using a new architecture called a Seq2Seq. Additionally, you can use pre-trained word embeddings to improve the performance of your model. Let's get started by following the steps below:

#### Step 1: Build your Vocabulary & create the Word Embeddings
The most important part of this step is to create your Vocabulary object using a corpus of data drawn from TorchText.
(Extra Credit)

Use Gensim to extract the word embeddings from one of its corpus'.
Use NLTK and Gensim to create a function to clean your text and look up the index of a word's embeddings.

#### Step 2: Create the Encoder
A Seq2Seq architecture consists of an encoder and a decoder unit. You will use Pytorch to build a full Seq2Seq model.
The first step of the architecture is to create an encoder with an LSTM unit.
(Extra Credit)

Load your pretrained embeddings into the LSTM unit.

#### Step 3: Create the Decoder
The second step of the architecture is to create a decoder using a second LSTM unit.

#### Step 4: Combine them into a Seq2Seq Architecture
To finalize your model, you will combine the encoder and decoder units into a working model.
The Seq2Seq2 model must be able to instantiate the encoder and decoder. Then, it will accept the inputs for these units and manage their interaction to get an output using the forward pass function.

#### Step 5: Train & evaluate your model
Finally you will train and evaluate your model using a Pytorch training loop.

#### Step 6: Interact with the Chatbot
Demonstrate your chatbot by converting the outputs of the model to text and displaying it's responses at the command line.