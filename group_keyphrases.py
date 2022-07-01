"""
    This file extracts a global list of keywords from some sub category in the arxiv snapshot of papers.
"""
import re
import json
import launch
import pickle

from tqdm import tqdm


arxiv_file = f"../../data/arxiv-metadata-oai-snapshot.json"
output_file = '../../data/csKeywordsNoDiv.json'


# Minimum frequency needed among papers for a keyword to be a candidate
filter_thresh = 0

# Arxiv category to filter by (in the below case, regex will match computer science category). See https://arxiv.org/category_taxonomy
filter_categ_re = re.compile(r"\bcs\.")

# The number of top keywords to extract per paper when computing global frequency counts
num_kwds_per_paper = 10


def get_paper_data():
    with open(arxiv_file, 'r') as f:
        for line in f:
            yield line


# Implementation of graph to keep track of co-occurences of papers
freq_dict = {}

def increment_counts(words):
    if words is None:
        return

    for word in words:
        if word not in freq_dict:
            freq_dict[word] = 1
        else:
            freq_dict[word] += 1



# Parse data from papers (keyword search)
embedding_distributor = launch.load_local_embedding_distributor()
pos_tagger = launch.load_local_corenlp_pos_tagger()

papers = get_paper_data()


# max_papers = 500
max_papers = None
p_i = 0

for i in tqdm(range(len(papers))):
    paper = papers[i]
    paper = json.loads(paper)
    cs_categs = filter_categ_re.search(paper['categories'])

    if cs_categs is not None:
        raw_text = paper['title'] + " " + paper['abstract']

        keywords_t = launch.extract_keyphrases(embedding_distributor, pos_tagger, raw_text, num_kwds_per_paper, 'en', beta=1)
        keywords = keywords_t[0]

        increment_counts(keywords)

    p_i += 1
    if max_papers is not None and p_i >= max_papers:
        break

    # if p_i % 10000 == 0:
    #     print("On " + str(p_i) + "th paper")



top_words = freq_dict.items()
top_words = filter(lambda x: x[1] > filter_thresh, top_words)
top_words = sorted(top_words, key=lambda x: x[1], reverse=True)




# Print top keywords and save results to file
# max_top_words = 400

# print("Top keywords for cs category: ")
# for i in range(min(max_top_words, len(top_words))):
#     word_t = top_words[i]
#     print("\t" + word_t[0] + ": " + str(word_t[1]))


with open(output_file, "w") as f:
    json.dump(top_words, f, indent=4)
