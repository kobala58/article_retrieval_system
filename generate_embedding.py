import argparse
import csv
import re

import pandas as pd
import spacy
from tqdm import tqdm
from spacy.lang.en import English
from sentence_transformers import SentenceTransformer


def setences_chunk_prep(sentences: list[str]) -> str:
    '''
    Join the sentences together into a paragraph-like structure
    :param sentences: list of string to merge together
    :return: Paragraph
    '''
    # joined_sentence_chunk = "".join(sentences).replace("  ", " ").replace('\n\n', '. ').strip()
    joined_sentence_chunk = "".join(sentences).replace("  ", " ").strip()
    joined_sentence_chunk = re.sub(r'\.([A-Z])', r'. \1',
                                   joined_sentence_chunk)  # ".A" -> ". A" for any full-stop/capital letter combo
    return joined_sentence_chunk


def split_large_chunk(sentences: list[str], max_size: int) -> list[list[str]]:
    '''

    :param sentences: List of sentences
    :param max_size: Max words count per chunk
    :return: List of lists containing sub chunks witch meet max_size criteria
    '''
    curr_word_cnt = 0
    sent_cnt = 0
    tmp = []
    splices = []
    while sent_cnt < len(sentences):
        sentence_len = len(sentences[sent_cnt].split(" "))
        if sentence_len > max_size:
            splices.append(sentences[sent_cnt])
            sent_cnt += 1
            continue
        while curr_word_cnt + len(sentences[sent_cnt].split(" ")) < max_size:
            tmp.append(sentences[sent_cnt])
            curr_word_cnt += len(sentences[sent_cnt].split(" "))
            sent_cnt += 1
            if sent_cnt == len(sentences):
                break
        splices.append(tmp)
        tmp = []
        curr_word_cnt = 0
    if len(tmp) != 0:
        splices.append(tmp)
    return splices


def split_list(inp_list: list, slice_size: int) -> list[list[str]]:
    '''
    Group sentences into chunks to further embedding

    :param list inp_list: List contains sentences that we want to merge into one chunk
    :param int slice_size: Number of sentences in one chunk
    :return: List of chunks (e.g. if we have 13 sentences and chunk size 5 we got list of chunks like [5, 5, 3])
    '''
    return [inp_list[i:i + slice_size] for i in range(0, len(inp_list), slice_size)]


def sensitizer(texts: list[dict], SENTENCE_CHUNK_SIZE: int, MIN_TOKENS: int, EMB_MODEL_MAX_WORDS: int) -> list[dict]:
    '''

    :param list[dict] texts: List of documents that we want to sentesize
    :param int SENTENCE_CHUNK_SIZE: Number of sentences in one chunk
    :return: list[dicts] containg new
    '''
    nlp = English()
    nlp.add_pipe("sentencizer")
    paragraphs = []
    for text in tqdm(texts, desc="Preparing texts"):
        text["sentences"] = list(nlp(text["Text"]).sents)
        text["sentences"] = [str(sentence) for sentence in text["sentences"]]  # turn npl internal object into str

        text["sentence_chunks"] = split_list(inp_list=text["sentences"], slice_size=SENTENCE_CHUNK_SIZE)
        # split into chunks

        text["chunks_num"] = len(text["sentence_chunks"])

        for sentence_chunk in text["sentence_chunks"]:
            chunk_dict = {"Title": text["Title"]}

            chunk_word_cnt = sum(len(chunk.split(" ")) for chunk in sentence_chunk)  # grab word count to check if 

            sentence_que = []

            if chunk_word_cnt > EMB_MODEL_MAX_WORDS:
                sentence_que.extend(split_large_chunk(sentence_chunk, EMB_MODEL_MAX_WORDS))
            else:
                sentence_que.append(sentence_chunk)

            for que_item in sentence_que:
                prepared_item = setences_chunk_prep(que_item)

                chunk_dict["sentence_chunk"] = prepared_item

                # Get stats about the chunk
                chunk_dict["chunk_char_count"] = len(prepared_item)
                chunk_dict["chunk_word_count"] = len([word for word in prepared_item.split(" ")])
                chunk_dict["chunk_token_count"] = len(prepared_item) / 4  # 1 token = ~4 characters
                if chunk_dict["chunk_token_count"] > MIN_TOKENS:
                    paragraphs.append(chunk_dict)
    print(f"Done. Generated {len(paragraphs)} chunks.")
    return paragraphs


def init_text_parser(filename: str) -> list[dict]:
    with open(filename, "r") as file:
        csv_file = csv.DictReader(file)
        return list(csv_file)


def embedder(text: list[dict], EMBEDDING_MODEL: str, ST_DEVICE: str) -> list[dict]:
    embedding_model = SentenceTransformer(model_name_or_path=EMBEDDING_MODEL,
                                          device=ST_DEVICE)
    embedding_model.to(ST_DEVICE)

    # Embeding
    for item in tqdm(text, desc="Encoding"):
        item["embedding"] = embedding_model.encode(item["sentence_chunk"])

    return text


def main(FILENAME: str, OUTFILENAME: str,
         SENTENCE_CHUNK_SIZE: int, EMBEDDING_MODEL: str, ST_DEVICE: str,
         EMB_MODEL_MAX_WORDS: int, MIN_TOKENS: int) -> None:
    """
    Steps: 
        1. Check if format is valid
        2. Split it into somewhat nice format
        2. Chunk it into selected size
        3. Embedded it using selected embedder from huggingface
        4. Return it into named csv
    """
    # first load named file
    # TODO: Make able to preload not only csv files
    if not FILENAME.endswith(".csv"):
        raise ValueError("File must be in csv format")

    init_df = pd.read_csv(FILENAME)  # load to dataframe

    if list(init_df.columns) != ["Title", "Text"]:  # check for proper column names
        # TODO: Make it select columns to further embeddings
        raise ValueError(f"File not containig proper columns. Excepted ['Title', 'Text'] found {init_df.columns}")

    texts = init_df.to_dict('records')
    texts = sensitizer(texts, SENTENCE_CHUNK_SIZE, MIN_TOKENS, EMB_MODEL_MAX_WORDS)
    encoded = embedder(texts, EMBEDDING_MODEL, ST_DEVICE)
    df = pd.DataFrame(encoded)
    df.to_csv(OUTFILENAME)
    print(f"Finished. Output file: {OUTFILENAME}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create csv with text embedding."
    )
    parser = argparse.ArgumentParser(description="Create csv with text embedding.")
    
    parser.add_argument("--filename", 
                        default="data.csv", 
                        type=str, 
                        help="Input CSV filename containing text data.")
    
    parser.add_argument("--outfilename", 
                        default="data_embedded_output.csv", 
                        type=str, 
                        help="Output CSV filename to save embedded text data.")
    
    parser.add_argument("--embedding_model", 
                        default="all-mpnet-base-v2", 
                        type=str, 
                        help="Name of the pre-trained embedding model to use.")
    
    parser.add_argument("--sentence_chunk_size", 
                        default=8, 
                        type=int, 
                        help="Maximum number of sentences to process at once.")
    
    parser.add_argument("--emb_model_max_words", 
                        default=354, 
                        type=int, 
                        help="Maximum number of words in a single input sequence to the embedding model.")
    
    parser.add_argument("--min_tokens", 
                        default=30, 
                        type=int, 
                        help="Minimum number of tokens required in a sentence to be considered for embedding.")
    
    parser.add_argument("--st_device", 
                        default="cpu", 
                        type=str, 
                        help="Device to perform the embedding computations on (e.g., 'cpu' or 'cuda:0').")
    
    parser.add_argument("--use_gpu", 
                        action="store_true", 
                        help="Flag to indicate whether to use GPU for computation.")
    args = parser.parse_args()

    main(args.filename, args.outfilename, args.sentence_chunk_size, args.embedding_model,
         args.st_device, args.emb_model_max_words, args.min_tokens)
