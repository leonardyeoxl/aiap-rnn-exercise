# AIAP RNN Exercise: Language Model
Exercise to practice building an RNN language model from scratch. You will train an RNN to read in characters from a text file and predict the next character, working with 40 characters at a time (independent of rest of the text). E.g. 
> Input: 'abcd' -> Label: 'bcde'
> Input: 'bcde' -> Label: 'cdef'
> ...
The model will learn the language and style of the text.  When we use the model to make predictions, we use the predicted output at each timestep as the input for the next timestep, to generate completely new text.

## Exercise Instructions:
* Create your own script or Jupyter Notebook following the 'Code Instructions' below.
* Use any machine learning library (e.g. TensorFlow, Keras). I used Keras here as it is extremely simple to put together.
* You can train on any of the text files in the 'data/raw' folder - or any other text file 

## Code Instructions:
1. Import numpy and your machine learning library of choice (e.g. Keras)
2. Open and read the .txt dataset of your choice in lower case. Create a list 'chars' of all the unique characters in the text, e.g. by creating a set of the text.  
3. Create a dictionary 'char_to_idx' that maps each unique character in the text to sequential integers, and a dictionary 'idx_to_char' that does the opposite. We'll use these dictionaries to convert our text into vectors for the RNN.
3. Create inputs and labels. 
	1. To create the inputs, we divide the text into sequences of length e.g. 40 characters. The labels are sequences of length 40 characters offset by one character.  E.g. Input: 'abcd' -> Label 'bcde', Input: 'bcde' -> Label: 'cdef', ...
	1. Convert inputs and labels into vectors using the 'char_to_idx' dictionary. Our desired vectors for each sequence will be a zero vector of size (sequence length, # characters), with a 1 in each row corresponding to the character's integer map ('one hot encoding'). We do this for all sequences to end up with input and output vectors of size (# sequences, sequence length, # characters)  
1. Build our model. 
	1. We want two LSTM cells of dimension 512, with the second LSTM cell taking as input the output of the first cell.  We want both cells to output sequences (i.e. we're interested in output at every timestep and not just at end).
	1. Add Dropout with keep probability 80% for the LSTM cells.
	1. Add a dense layer of dimension size equal to the number of unique characters in your text. This layer converts the LSTM output of dimension 512 into the odds that the output should be each character (e.g. 'c', 'k')
	1. Apply softmax to the dense layer to convert the values into probabilities the output should be each character.
	1. Calculate loss as the categorical cross entropy between the predicted softmax probabilities and labels.
	1. Apply an optimizer e.g. RMSProp
7. Fit the model on CPU or GPU of choice for 1 Epoch. Use batchsize of 128.  
8. To generate predictions, create a seed string of several characters as starting input. Then create a for loop that predicts on this input (take the arg max of the softmax output to select the predicted character), appends the prediction to the input, and repeat the loop.  Ensure the ending state at each step is passed as input for the next step.   

## Example Output
The model will learn to create english words in a few batches. By 1 Epoch, you will have relatively coherent sequences. 1 Epoch training Shakespeare on GPU took ~1 hr reaching loss ~1.25, producing predictions like the following:  
```
brutus:
i will not do thee worse.

proteus:
what shall we do? i will not stay to hear it.

pandarus:
i will seek him to his companies. i will tell you what i say,
that i might see the sea and land of the streets,
and the substance of the sea was made to the sea,
and the device of the world, i would not have it so.

don pedro:
```
## Sources and Further Reference
* Torch - Unreasonable effectiveness of recurrent neural networks http://karpathy.github.io/2015/05/21/rnn-effectiveness/
* Keras - LSTM text generation https://github.com/keras-team/keras/blob/master/examples/lstm_text_generation.py
* Tensorflow - Shakespeare https://github.com/martin-gorner/tensorflow-rnn-shakespeare
Data sources:
* Sherlock holmes text https://sherlock-holm.es/ascii/
* Shakespeare https://github.com/karpathy/char-rnn
