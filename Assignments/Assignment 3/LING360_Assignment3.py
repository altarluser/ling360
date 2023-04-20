# -*- coding: utf-8 -*-
"""AhmetAltarLuser_Assignment3.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1pLDbeN1gXJNT2nykuEmJOYqb6F3a8_4E

# Assignment 3

This is <font color='cyan'>Assignment 3</font> for the LING360 - Computational Methods in Lingustics course and it is worth a total of  <font color='cyan'>**5 points**</font>.
The assignment covers the basic knowledge of Python. 

The topics include:
1. For loops
1. Regular Expressions
1. Control Flow
1. Functions


There's a total of  <font color='cyan'>**3 main tasks**</font>. For each task, please write your code between the following lines:

```
## YOUR CODE STARTS



## YOUR CODE ENDS
```

Before working on the assignment, please copy this notebook to your own drive. You can use ```Save a copy in Drive``` under the ```File``` menu on top left.

Please, run every cell in your code to make sure that it works properly before submitting it. 

Once you are ready to submit, download two versions of your code:

*   Download .ipynb
*   Download .py

These are both available under the ```File``` menu on top left. 

Then, compress your files (zip, rar, or whatever) and upload the compressed file to Moodle.

If you have any questions, please contact karahan.sahin@boun.edu.tr
"""

## DO NOT EDIT THE CODE BELOW

# RUN THIS LINE FIRST !!
import re
import requests
!pip install stanza --quiet
import stanza
nlp = stanza.Pipeline(lang='tr', 
                      processors='tokenize,lemma', 
                      lemma_pretagged=True, 
                      tokenize_pretokenized=True, 
                      use_gpu=False)

def tokenize(sentence): return [token["text"] for token in nlp(sentence).sentences[0].to_dict()]
def lemmatize(sentence): return [token["lemma"] for token in nlp(sentence).sentences[0].to_dict()]

## DO NOT EDIT THE CODE ABOVE

# The code below gets a Turkish lexicon from the Zemberek project. This lexicon
# will be used as a lemma list. 
# To better understand what the code is doing, you can go to the link below to 
# see what the file looks like.

## DO NOT EDIT THE CODE BELOW

# RUN THIS LINE FIRST !!
file = requests.get("https://raw.githubusercontent.com/Loodos/zemberek-python/master/zemberek/resources/lexicon.csv")

# You only need to run this code as well
lemma_list = sorted({line.split("\t")[2] for line in file.text.split("\n")[:-1]
                        if not (line.split("\t")[0].endswith("Prop"))
                        and len(line.split("\t")[2]) > 2 })

print("Lemma list is extracted")

## DO NOT EDIT THE CODE ABOVE

# The code below creates a dictionary where the keys are lemmas and their values
# are POS (parts of speech) tags (e.g. Adj, Noun, Verb, etc.). 

## DO NOT EDIT THE CODE BELOW

# RUN THIS LINE FIRST !!
word_to_pos_dictionary = {line.split("\t")[2]: line.split("\t")[3]  
                              for line in file.text.split("\n")[:-1]}

## DO NOT EDIT THE CODE ABOVE

"""## <font color='#FFBD33'>**Q1:** Tokenizer</font> `1.5 points`

Implement a `tokenize()` function which takes string argument called `sentence` and returns the tokens of a given sentence. 

<font color='#FFBD33'>**Instructions:**</font>

1. First, remove all punctuation using `re.sub()`.
2. Then, split the string on whitespace using the function we built in LAB 3.
3. Create an empty list and assign it to a variable named `tokens`.
4. Using a `for loop`, append only the strings that consist of alphabetic characters to the `tokens` list.
5. Finally return the `tokens` list.

<font color='#FFBD33'>**Notes:**</font>
1. Watch out! Some punctuations are escape characters, for those cases, add the `\` character before the punctuation character.
1. To find whether a character is composed of only alphabetic, use the `"your_string".isalpha()` method.

"""

## YOUR CODE STARTS

def tokenize(sentence):
  #no_punc = re.sub(r'''[!"#\$%'()+,-./:;?[]{}£¥©«®»'’".\.\.]+''', " ", sentence)
  no_punc = re.sub(r"[^\w\s]+"," ", sentence)
  splitted_sentence = no_punc.split(" ")
  tokens = []
  
  for token in splitted_sentence:
    if token.isalpha():
     tokens.append(token)
  return tokens
## YOUR CODE ENDS

"""## <font color='#FFBD33'>**Q2:** Lemmatizer</font> `2 points`
Write a function called `lemmatize()` which takes a string argument named `sentence`. You need to first tokenize the sentence and then lemmatize. 

If you completed the previous task successfully, then use your `tokenize()` function to tokenize the sentence. 

If you could not successfully implement a `tokenize()` function, then use the `tokenize()` function we provide below. 


If you were not able to write your `tokenize()` function properly, you can use the pre-defined function in first cell. Just rerun the first cell in this notebook.

<font color='#FFBD33'>**Instructions:**</font>
1. First, make sure that you run the cell above where the  `lemma_list` is defined.
2. After you define your function, create an empty list and assign it to a variable named `lemmas`.
3. Create a list called `tokens` which holds the tokens from the tokenized sentence via `tokenize()` function.
4. Using a for loop iterating `tokens` list, check if the token starts with any lemma found in `lemma_list` via `re.findall()`. Your `regex` should be in format:
    ```
    "^(araba|al|ak|.....)"
    ```
5. If you find any lemmas, add the lemma to the `lemmas` list.
6. Finally return the `lemmas` list.


<font color='#FFBD33'>**Notes:**</font>
1. Don't worry! Some lemmas can be missing! We have provided some test sentences below but check the regex results before appending to the `lemmas` list.
1. To write a regex, you need to create string which finds a string start with any of the lemmas. Since you cannot mannualy add each lemma, use `"".join(your_list)` method to automatically concatenate all lemmas as the example below.

    ```python
    ",".join(["a","b","c"])
    # Output: "a,b,c"

    ":)".join(["a","b","c"])
    # Output: "a:)b:)c"
    ```
"""

## YOUR CODE STARTS
def lemmatize(sentence):
  lemmas = []
  tokens = tokenize(sentence)
  for token in tokens:
    matches = re.findall(r'^(?:' + '|'.join(lemma_list) + r')', token)
    if matches:
      lemmas.append(matches[0])
  return lemmas

## YOUR CODE ENDS

lemmatize("ben geldim ama sen gelmedin") == ['ben', 'gel', 'ama', 'sen', 'gel']

lemmatize("bizler hiç bilmezdik ama neler olmuştu") == ['biz', 'hiç', 'bil', 'ama', 'nel', 'olm']

"""## <font color='#FFBD33'>**Q3:** Part-of-Speech Statistics</font> `1.5 points`

Calculate total count of part-of-speech of tokens in `sentences` list, via using `word_to_pos_dictionary` dictionary. For each sentence, first tokenize, then lemmatize using the corresponding functions to find the part-of-speech of the word. After that increase the number of the part-of-speech in the dictionary `YOUR_POS`. If you were not able to write your `tokenize()` and `lemmatize()` functions, you can use the pre-defined function in the first cell. 

<font color='#FFBD33'>**Instructions:**</font>
1. Write a for loop which iterates over each sentence in `sentences`.
2. First lemmatize the string with `lemmatize()` function using either your custom function or lemmatize function we defined.
3. Then increment the number count of part-of-speech tag, aka `POS` tag, for each of the extracted lemmas by getting the `POS` tag of lemma from `word_to_pos_dictionary`.
"""

print("Part-of-speech of word 'sev' is:", word_to_pos_dictionary["sev"])
print("Part-of-speech of word 'bilgi' is:", word_to_pos_dictionary["bilgi"])

sentences = [
 'ben geldim',
 'gördüm',
 'ama o daha gelmemişti.',
 'sen biliyorsun...',
 'onlar anlamazlar !',
 'nasılsınız?',
 'iyiyiz"',
]

YOUR_POS_COUNT = {
    'Adj': 0,
    'Adv': 0,
    'Conj': 0,
    'Det': 0,
    'Dup': 0,
    'Interj': 0,
    'Noun': 0,
    'Num': 0,
    'Postp': 0,
    'Pron': 0,
    'Punc': 0,
    'Ques': 0,
    'Verb': 0
}

# YOUR CODE STARTS
for sentence in sentences:
  lemmatized = lemmatize(sentence)
  for lemma in lemmatized:
    if lemma in word_to_pos_dictionary:
      pos = word_to_pos_dictionary[lemma]
      YOUR_POS_COUNT[pos] += 1

print(YOUR_POS_COUNT)

# YOUR CODE ENDS