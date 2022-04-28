import sys
sys.path.insert(1, '/Users/ashutoshukey/Downloads/Forward_Data_Lab/Code/misc')
from get_papers import mag_get_papers, mag_get_author

import launch


def get_author_abbrev(full_name):
    name_parts = author_name.split(" ")
    author_abbrev = "".join(map(lambda x: x[0].lower(), name_parts[:-1]))
    author_abbrev += name_parts[-1].lower()

    return author_abbrev


# Keep track of word counts
freq_dict = {}

def increment_counts(words):
    if words is None:
        return

    for word in words:
        if word not in freq_dict:
            freq_dict[word] = 1
        else:
            freq_dict[word] += 1



author_name = 'Jiawei Han'
author_institution = 'Microsoft'
author_abbrev = get_author_abbrev(author_name)

output_file = '../../data/Researcher_Keywords/' + author_abbrev + 'Keywords.csv'

num_requested_papers = 5000

author = mag_get_author(author_name, author_institution)
papers = mag_get_papers(author['id'], num_requested_papers)
print('Obtained ' + str(len(papers)) + ' abstracts.')


# Extract keywords from abstracts
embedding_distributor = launch.load_local_embedding_distributor()
pos_tagger = launch.load_local_corenlp_pos_tagger()


print("Starting paper keyword extraction: ")
p_i = 0
for paper in papers:
    # raw_text = paper['title'] + ' ' + paper['abstract']
    raw_text = paper['abstract']

    keywords_t = launch.extract_keyphrases(embedding_distributor, pos_tagger, raw_text, 10, 'en', beta=1)
    keywords = keywords_t[0]

    increment_counts(keywords)

    p_i += 1
    if p_i % 100 == 0:
        print("On " + str(p_i) + "th abstract...")


filter_thresh = 1
top_words = freq_dict.items()

print("Processing and writing data...")
max_words = min(400, len(top_words))
top_words = filter(lambda x: x[1] > filter_thresh, top_words)
top_words = sorted(top_words, key=lambda x: x[1], reverse=True)[:max_words]

with open(output_file, "w") as f:
    for word_t in top_words:
        f.write(word_t[0] + "," + str(word_t[1]) +"\n")

print("Done.")
