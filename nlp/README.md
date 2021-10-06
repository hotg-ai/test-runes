Steps to install `rune`

1. Build [librunecoral](https://github.com/hotg-ai/librunecoral)
2. Clone rune repo: `git clone https://github.com/hotg-ai/rune.git`
3. `cd rune`
4. `cargo install --force --path ./crates/rune-cli`


Steps to run `Bert model`:
1. `git clone https://github.com/hotg-ai/test-runes.git`
2. `cd test-runes`
3. `git checkout 21b4c6e28afb14bef0a82525222bc53ec55742e4`
4. `cd nlp/bert`
5. Download [mobilebert.tflite](https://tfhub.dev/tensorflow/lite-model/mobilebert/1/default/1) (file is above 100Mb so with git clone you will only get the text file)
6. Generate `bert.rune` with this command: `rune build Runefile.yml --unstable --rune-repo-dir ~/rune`

      or 

  To avoid previous steps (1-6) download `bert.rune` from [here](https://drive.google.com/file/d/18xxcXX9SlNgx9Tc6q2cmL7yF-HymuE9w/view?usp=sharing)


Steps to Run the model:

There two files `input1.txt` and `input2.txt`. The `input2.txt` file contains the context from where the model will find answers to question you will pass in the `input1.txt`.
To run inference: `rune run bert.rune --raw input1.txt input2.txt`. You can change the question and context according to you convenience.
