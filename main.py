import pyarrow.parquet as pq
from tokenizer import HindiTokenizer, VOCAB_SIZE

num_merges = VOCAB_SIZE - 257

# dataset -> https://huggingface.co/datasets/wikimedia/wikipedia/blob/main/20231101.hi/train-00001-of-00002.parquet

files = ['~/Downloads/train-00001-of-00002.parquet']
# Read the Parquet file
columns_to_read = ['text']
texts = ''
init_tokens = []
perc = 20
for f in files:
    df = pq.ParquetFile(f)

    # Display the first few rows
    for i in range(0, int(perc*0.01 * df.num_row_groups)):
        # Read one row group (chunk) at a time
        df_chunk = df.read_row_group(i, columns=columns_to_read).to_pandas()

        # Process the chunk
        # print(f"Processing chunk {i + 1}/{df.num_row_groups}")

        chunk = df_chunk.to_string()
        texts += chunk + '\n'

        #if i == 1:
        #    break

    #break

'''
texts = """मेरा भारत महान।
    १२३४५ का गणितीय महत्व।
    भारत एक महान देश है॥"""
'''

#print(len(texts))
tokenizer = HindiTokenizer()
tokenizer.learn_bpe_vocab(texts, num_merges=num_merges)

# Save the learned vocabulary
tokenizer.save_bpe_vocab("hindi_bpe_vocab.model")
#tokenizer.load_bpe_vocab("hindi_bpe_vocab.model")

test_text = "मेरा भारत महान। १२३४५॥ Tiger's dead. I'VE killed it!"
#print(tokenizer.decode([256, 149, 259, 302]))
print(test_text == tokenizer.decode(tokenizer.encode(test_text)))