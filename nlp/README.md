First you have to install the [rune](https://hotg.dev/docs/)

After installing rune follow these steps to run `Bert model`:
1. `git clone https://github.com/hotg-ai/test-runes.git`
2. `cd test-runes`
3. `cd nlp/bert`
4. Download [mobilebert.tflite](https://tfhub.dev/tensorflow/lite-model/mobilebert/1/default/1) (file is above 100Mb so with git clone you will only get the text file)
5. Generate `bert.rune` with this command: `rune build Runefile.yml`

      or 

  To avoid previous steps (1-5) download `bert.rune` from [here](https://raw.githubusercontent.com/hotg-ai/test-runes/master/nlp/bert/bert.rune)


Steps to Run the model:

There two files `input1.txt` and `input2.txt`. The `input2.txt` file contains the context from where the model will find answers to question you will pass in the `input1.txt`.
To run inference: `rune run bert.rune --raw input1.txt input2.txt`. You can change the question and context according to you convenience.
