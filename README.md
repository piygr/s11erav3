# HindiTokenizer Overview

This repository provides a Hindi Tokenizer implemented in Python, leveraging Byte Pair Encoding (BPE) for efficient subword tokenization. It is designed to handle Hindi text data effectively and can be integrated into NLP pipelines requiring subword tokenization.

## Features

- **Subword Tokenization**: Efficiently breaks text into subword units to manage out-of-vocabulary words.
- **Pre-trained Vocabulary**: Includes a pre-trained BPE vocabulary model for Hindi.
- **Easy-to-Use Interface**: Provides simple methods for tokenizing and detokenizing Hindi text.

## Prerequisites

- Python 3.6+

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/piygr/s11erav3.git
   cd s11erav3
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Import the Tokenizer**:
   ```python
   from tokenizer import HindiTokenizer
   ```

2. **Initialize the Tokenizer**:
   ```python
   tokenizer = HindiTokenizer.load_bpe_vocab("hindi_bpe_vocab.model")
   ```

3. **Tokenize Hindi Text**:
   ```python
   text = "आपका स्वागत है"
   tokens = tokenizer.encode(text)
   print(tokens)  # Output: List of tokens
   ```

4. **Detokenize Tokens**:
   ```python
   detokenized_text = tokenizer.decode(tokens)
   print(detokenized_text)  # Output: Original Hindi text
   ```

## File Overview

- **`tokenizer.py`**: Contains the `HindiTokenizer` class with methods for tokenization and detokenization.
- **`hindi_bpe_vocab.model`**: Pre-trained Byte Pair Encoding (BPE) vocabulary model for Hindi text.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- This tokenizer was trained on the [Wikipedia dataset](https://huggingface.co/datasets/wikimedia/wikipedia/blob/main/20231101.hi/train-00001-of-00002.parquet) for subword tokenization.

## Contact

For any queries or issues, please create an issue in the repository or reach out to the maintainers.
