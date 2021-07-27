from tensorflow.keras.preprocessing.text import Tokenizer  #Helps split text into words/ or tokens
from tensorflow.keras.preprocessing.sequence import pad_sequences

sentences = ["First sentence to tokenize",
             "I love Dogs",
             "Dogs are Crazy",
             "Go team Cats"
             ]

#  Trying to tokenize these sentences using the Tokenizer class

#  Creating an instance of Tokenizer
tokenizer = Tokenizer(num_words=100,
                       oov_token='<OOV>')

tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
print(word_index)

#  Converting texts to sequences
sequences = tokenizer.texts_to_sequences(sentences)
print(sequences)

#  Padding Sequences to, this will be useful while building models, because all inputs should be of the same size

padded_sequences = pad_sequences(sequences, maxlen=4, padding='post', truncating='pre')
#  There are 3 important parameters for this function to keep in mind
#  max_len determines the length of the sequence
#  padding if the text should be padded in the beginning or the end
#  truncating depending on which part holds more context - from the front or the back

print(padded_sequences)

