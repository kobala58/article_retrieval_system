# Article Retrieval System


## Embeddings generator
File `generate_embedding.py` contains solution to generate new csv with embeddings attached. 

### Instalation
	pip install -r requirements.txt
	

### Usage 
	python generate_embedding.py
	
###  Flags aviable

1. `--filename`
   - **Default Value:** `"data.csv"`
   - **Type:** String
   - **Description:** Name of the CSV file containing the text data to embed.

2. `--outfilename`
   - **Default Value:** `"data_embedded_output.csv"`
   - **Type:** String
   - **Description:** Name of the CSV file to save the embedded text data.

3. `--embedding_model`
   - **Default Value:** `"all-mpnet-base-v2"`
   - **Type:** String
   - **Description:** Name of the text embedding model to be used for processing.

4. `--sentence_chunk_size`
   - **Default Value:** `8`
   - **Type:** Integer
   - **Description:** Maximum number of sentences to process at once.

5. `--emb_model_max_words`
   - **Default Value:** `354`
   - **Type:** Integer
   - **Description:** Maximum number of words in a single input sequence to the embedding model.

6. `--min_tokens`
   - **Default Value:** `30`
   - **Type:** Integer
   - **Description:** Minimum number of tokens required in a sentence to be considered for embedding.

7. `--st_device`
   - **Default Value:** `"cpu"`
   - **Type:** String
   - **Description:** Device on which the embedding computations will be performed (e.g., `"cpu"` or `"cuda:0"`).

8. `--use_gpu`
   - **Type:** Flag
   - **Description:** Flag indicating whether to use a GPU for computation.

## API 

### Instalation
	cd api
	docker build -t rag_api .
	docker run -d -p 8000:8000 rag_api
### Usage
Head to http://localhost:8000/docs to see endpoint returning relevant articles.