use core::str::FromStr;
use tokenizers::tokenizer::{BertTokenizer, Tokenizer, TruncationStrategy};
use tokenizers::vocab::{BertVocab, Vocab};

fn main() {
    // let strip_accents = false;
    // let lower_case = false;
    // let vocab = BertVocab::from_file("bert-base-uncased-vocab.txt").unwrap();

    // let bert_tokenizer = BertTokenizer::from_existing_vocab(vocab, lower_case, strip_accents);

    let vocabulary_text = include_str!("bert-base-uncased-vocab.txt");

    let vocab = BertVocab::from_str(vocabulary_text).unwrap();

    let bert_tokenizer = BertTokenizer::from_existing_vocab(vocab, false, false);

    let test_sentence = ("After stealing money from the bank vault. The bank robber was seen fishing on the Mississippi river bank.").to_string();
    let token = bert_tokenizer.encode(
        &test_sentence,
        None,
        128,
        &TruncationStrategy::LongestFirst,
        0,
    );

    println!("\ntoken_ids: {:?}\n", token);
}
