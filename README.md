# AIAP RNN Exercise: Language Model
Exercise to practice building an RNN language model from scratch. You will train an RNN to read in characters from a text file and predict the next character, working with sequences of e.g. 40 characters at a time (independent of rest of the text). This will look like:

> Input: 'abcd' -> Label: 'bcde'

> Input: 'bcde' -> Label: 'cdef'

> ...

The model will learn the language and style of the text.  When we use the model to make predictions, we feed back the predicted output at each timestep as the input to the next timestep, to generate completely new text.

## Exercise Instructions:
* Create your own script or Jupyter Notebook following the 'Code Instructions' below.
* Use any machine learning library (e.g. TensorFlow, Keras). I used Keras here as it is extremely simple to put together.
* You can practice using the text files in the 'data/raw' folder as data. Find another dataset (text or even code or music) for your actual model. 

## Code Instructions:
1. Import numpy and your machine learning library of choice (e.g. Keras)
2. Open and read the .txt dataset of your choice in lower case. Create a list 'chars' of all the unique characters in the text, e.g. by creating a set of the text.
3. Create a dictionary 'char_to_idx' that maps each unique character (including spaces, punctuation, etc.) in the text to integers (starting from 0 and increasing), and a dictionary 'idx_to_char' that does the opposite. We'll use these dictionaries to convert our text into vectors for the RNN.
3. Create inputs and labels. 
	1. To create the inputs, we divide the text into sequences of fixed length (e.g. 40 characters). The labels are sequences of the same length offset by one character.  E.g. Input: 'abcd' -> Label 'bcde', Input: 'bcde' -> Label: 'cdef', ...
	1. Convert inputs and labels into vectors using the 'char_to_idx' dictionary. Our desired vectors for each sequence will be a zero vector of size (sequence length, # characters), with a 1 in each row corresponding to the character's corresponding integer ('one hot encoding'). We do this for all sequences to end up with input and output vectors of size (# sequences, sequence length, # characters)  
1. Build our model. 
	1. We want two LSTM cells of dimension 512, with the second LSTM cell taking as input the output of the first cell.  We want both cells to output sequences (i.e. we're interested in output at every timestep and not just at the end).
	1. Add Dropout with keep probability 80% for the LSTM cells.
	1. Add a dense layer of dimension size equal to the number of unique characters in your text. This layer converts the LSTM output of dimension 512 into the odds that the output should be each character (e.g. 'c' or 'k')
	1. Apply softmax to the dense layer to convert the values into probabilities the output should be each character.
	1. Calculate categorical cross entropy loss between the predicted softmax probabilities and labels.
	1. Apply an optimizer e.g. RMSProp with an appropriate learning rate
7. Fit the model on CPU or GPU of choice for 1 Epoch. You can use batchsize of e.g. 128. 
8. To generate predictions, create a seed string of several characters as starting input. Then create a for loop that predicts on this input (take the 'arg max' of the softmax output to select the predicted character), appends the prediction to the input, and repeats.  Ensure the ending state at each step is passed as an input to the next step.   

## Example Output
The model will learn to create coherent English words in a few batches. By 1 Epoch, you will have relatively coherent sequences of words. 1 Epoch training Shakespeare on GPU took ~1 hr reaching loss ~1.25, producing predictions like the following:  
```
emperor:
i have seen her to seek the strength of the streets,
and the substance of the sea was made to the sea,
and the device of the world, i would not have it so.

don pedro:
what say you to this?

cassio:
i pray you, sir, the duke of york is strange.

antipholus of ephesus:
i will not stay to see him so 
```
## Sources and Further Reference
* Withdrawn until Thursday
