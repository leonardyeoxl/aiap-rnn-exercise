# AIAP RNN Exercise
Exercise for you to practice building an RNN language model from scratch. This exercise trains...  You should create your own script or Jupyter Notebook following the 'Code Instructions' below.

## Notes:
* You can follow this guide using any machine learning library (e.g. TensorFlow, Keras). I used Keras here as it is very simple to put the pieces together.
* Data...

## Code Instructions:
1. Import numpy and whatever machine learning library you are using (e.g. Keras)
2. Open and read the .txt dataset of your choice in lower case. Create a list 'chars' of all the unique characters in the text, e.g. by creating a set of the text.  
3. Create a dictionary 'char_to_ix' that maps each unique character in the text to an integer, and a dictionary 'ix_to_char' that does the opposite. We'll use these dictionaries to convert our text into vectors for the RNN.
4. Create our inputs and labels. To create the inputs, we divide the text into chunks of length 40 characters. The labels are chunks of length 40 characters offset by one character.  E.g. Input: 'abcd' -> Label 'bcde', Input: 'bcde' -> Label: 'cdef', ...
5. Convert inputs and labels into vectors using the 'char_to_ix' dictionary.  ...
6. Build our model. 
	1. We want two LSTM cells of dimension 512, with the second LSTM cell taking as input the output of the first cell.  We want both cells to output sequences (i.e. we're interested in output at every timestep and not just at end).
	1. Add Dropout with keep probability 80% for the LSTM cells.
	1. Add a dense layer of dimension size equal to the number of unique characters in your text. This layer converts the LSTM output of dimension 512 into the odds that the output should be each character (e.g. 'c', 'k')
	1. Apply softmax to the dense layer to convert the values into probabilities the output should be each character.
	1. Calculate loss as the categorical cross entropy between the predicted softmax probabilities and labels.
	1. Apply an optimizer e.g. RMSProp
7. Fit the model. 

