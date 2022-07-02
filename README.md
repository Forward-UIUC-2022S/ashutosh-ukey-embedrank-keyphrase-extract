This module is responsible for extracting a global list of keywords from a set of papers. This repo is a fork of https://github.com/swisscom/ai-research-keyphrase-extraction.

# Setup

## Local

1. Download full Stanford CoreNLP Tagger version 3.8.0
http://nlp.stanford.edu/software/stanford-corenlp-full-2018-02-27.zip

2. Download arxiv papers snapshot from https://www.kaggle.com/datasets/Cornell-University/arxiv (the arxiv-metadata-oai-snapshot.json file)

2. Install sent2vec from 
https://github.com/epfml/sent2vec
    * Clone/download the directory
    * go to sent2vec directory
    * git checkout f827d014a473aa22b2fef28d9e29211d50808d48
    * make
    * pip install cython
    * inside the src folder 
        * ``python setup.py build_ext``
        * ``pip install . ``
        * (In OSX) If the setup.py throws an **error** (ignore warnings), open setup.py and add '-stdlib=libc++' in the compile_opts list.        
    * Download sent2vec pretrained model wiki_bigrams.bin (see readme of Sent2Vec repo, should be ~16GB). 
     
3. Install requirements
    
    After cloning this repository go to the root directory and
    ``pip install -r requirements.txt``

5. Download NLTK data
```
import nltk 
nltk.download('punkt')
```

6. Launch Stanford Core NLP tagger
    * Open a new terminal
    * Go to the stanford-core-nlp-full directory
    * Run the server `java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -preload tokenize,ssplit,pos -status_port 9000 -port 9000 -timeout 15000 & `


6. Set the paths in config.ini.template
    * You can leave [STANFORDTAGGER] parameters empty
    * For [STANFORDCORENLPTAGGER] :
        * set host to localhost
        * set port to 9000
    * For [SENT2VEC]:
        * set your model_path to the pretrained model
        your_path_to_model/wiki_bigrams.bin (if you choosed wiki_bigrams.bin)
    * rename config.ini.template to config.ini

## Docker

See README of original repo for instructions on how to setup using Docker.

&nbsp;
# Usage (Functional Design)

Make sure the CoreNLP server is running

## Set of Papers
To extract keywords from a corpus of papers (e.g. arxiv), run the file.

```bash
python3 group_keyphrases.py
```

Make sure you specify `arxiv_file` and `output_file` variables. Here is a description of other program hyper-parameters:
- filter_categ_re: regular expression of subcategory of arxiv topics to filter by
- num_kwds_per_paper: number of top keywords to extract from each paper
- filter_thresh: minimum frequency needed among papers for a keyword to be a candidate

The output file will be a json 

&nbsp;
## Single Paper

The above program relies on being able to extract keywords from a single paper:

```
import launch

embedding_distributor = launch.load_local_embedding_distributor()
pos_tagger = launch.load_local_corenlp_pos_tagger()

kp1 = launch.extract_keyphrases(embedding_distributor, pos_tagger, raw_text, 10, 'en')  #extract 10 keyphrases
kp2 = launch.extract_keyphrases(embedding_distributor, pos_tagger, raw_text2, 10, 'en')
...
```

This return for each text a tuple containing three lists:
1) The top N candidates (string) i.e keyphrases
2) For each keyphrase the associated relevance score
3) For each keyphrase a list of alias (other candidates very similar to the one selected
as keyphrase)

&nbsp;
# Methodology (Algorithmic Design)

## Single paper
By using sentence embeddings , EmbedRank embeds both the document and candidate phrases into the same embedding space.

![embedrank](embedrank.gif)

N candidates are selected as keyphrases by using Maximal Margin Relevance using the cosine similarity between the candidates and the
document in order to model the informativness and the cosine
similarity between the candidates is used to model the diversity.

An hyperparameter, beta (default=0.55), controls the importance given to 
informativness and diversity when extracting keyphrases.
(beta = 1 only informativness , beta = 0 only diversity)
You can change the beta hyperparameter value when calling extract_keyphrases:

```
kp1 = launch.extract_keyphrases(embedding_distributor, pos_tagger, raw_text, 10, 'en', beta=0.8)  #extract 10 keyphrases with beta=0.8

```

If you want to replicate the results of the paper you have to set beta to 1 or 0.5 and turn off the alias feature by specifiying alias_threshold=1 to extract_keyphrases method.

&nbsp;
## Multiple papers

Using the above algorithm with `beta=1`, we extract the top `num_kwds_per_paper` (the default is 10) from each paper and aggregate the number of papers each keyword appears in. We sort the keywords by this frequency from most to least frequent.

To improve keyword quality, users may also take the intersection of these keywords with some other set of keywords.

