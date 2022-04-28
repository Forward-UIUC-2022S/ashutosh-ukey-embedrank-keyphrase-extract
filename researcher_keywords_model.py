import sys
import math
import json
import pickle
import launch
sys.path.insert(1, '/Users/ashutoshukey/Downloads/Forward_Data_Lab/Code/misc')
from get_papers import mag_get_papers, mag_get_author
from explore_trie import get_word_group

data_root_dir = '../../data/'
keywords_file = data_root_dir + 'csKeywordsNorm.json'
group_idx_file = data_root_dir + 'new_kwgroups.pickle'
rev_group_idx_file = data_root_dir + 'rev_kwgroups.pickle'
cs_freqs_file = data_root_dir + 'new_freqs.pickle'


def get_author_abbrev(full_name):
    name_parts = full_name.split(" ")
    author_abbrev = "".join(map(lambda x: x[0].lower(), name_parts[:-1]))
    author_abbrev += name_parts[-1].lower()

    return author_abbrev


with open(cs_freqs_file, 'rb') as f:
    cs_keyword_freqs = pickle.load(f)

# Keep track of word counts
freq_dict = {}

def increment_scores(word_ts, paper):
    words = word_ts[0]
    num_words = len(words)

    if words is None:
        return

    year_score = ((paper['year'] - 1970) ** 2) / 3000
    paper_multiplier = year_score * paper['num_citations'] / paper['author']['order']


    total_score = 0
    for w_i in range(num_words):
        word = words[w_i]

        # if word in cs_keyword_freqs and cs_keyword_freqs[word] > 1000:
        #     word_ts[1][w_i] /= cs_keyword_freqs[word]

        total_score += word_ts[1][w_i]


    for w_i in range(num_words):
        word = words[w_i]
        word_score = word_ts[1][w_i] * paper_multiplier

        if word not in freq_dict:
            freq_dict[word] = word_score
        else:
            freq_dict[word] += word_score



# Extract keywords from abstracts
embedding_distributor = launch.load_local_embedding_distributor()
pos_tagger = launch.load_local_corenlp_pos_tagger()


with open(keywords_file) as f:
    keywords = json.load(f)
    num_search_limit = min(75000, len(keywords))
    keywords_set = set(map(lambda x: x[0], keywords[:num_search_limit]))


with open(group_idx_file, 'rb') as f:
    word_to_group = pickle.load(f)


with open(rev_group_idx_file, 'rb') as f:
    word_to_group_rev = pickle.load(f)


def profile_researcher(author_name, author_institution):
    author_abbrev = get_author_abbrev(author_name)

    output_file = data_root_dir + 'Researcher_Keywords/' + author_abbrev + 'Keywords-uniq4.csv'

    num_requested_papers = 5000

    author = mag_get_author(author_name, author_institution)
    papers = mag_get_papers(author['id'], num_requested_papers)
    print('Obtained ' + str(len(papers)) + ' abstracts.')


    print("Starting paper keyword extraction: ")
    p_i = 0
    for paper in papers:
        # raw_text = paper['title'] + ' ' + paper['abstract']
        raw_text = paper['abstract']

        keywords_t = launch.extract_keyphrases(embedding_distributor, pos_tagger, raw_text, 10, 'en', beta=1)

        # new_keywords_t = [[], []]
        # for i in range(len(keywords_t[0])):
        #     word = keywords_t[0][i]
        #     if word in cs_keyword_freqs:
        #         new_keywords_t[0].append(word)
        #         new_keywords_t[1].append(keywords_t[1][i])
        #
        # # keywords = keywords_t[0]
        # keywords_t = new_keywords_t

        increment_scores(keywords_t, paper)

        p_i += 1
        if p_i % 100 == 0:
            print("On " + str(p_i) + "th abstract...")


    # Filter out general keywords
    for word in freq_dict:
        if word in cs_keyword_freqs and cs_keyword_freqs[word] > 1000:
            freq_dict[word] /= cs_keyword_freqs[word]

    top_words = freq_dict.items()


    max_words = min(200, len(top_words))
    top_words = filter(lambda x: x[0] in keywords_set, top_words)
    top_words = sorted(top_words, key=lambda x: x[1], reverse=True)[:max_words]


    # Ignore similar keywords
    curr_groups = set()
    curr_rev_groups = set()

    unique_top_words = []
    for word_t in top_words:
        word = word_t[0]
        w_group = get_word_group(word, word_to_group)
        w_group_rev = get_word_group(word, word_to_group_rev)

        if w_group == -1:
            print("Unable to find word: '" +  word + "'")

        elif w_group not in curr_groups and w_group_rev not in curr_rev_groups:
        # elif w_group not in curr_groups:
            curr_groups.add(w_group)
            curr_rev_groups.add(w_group_rev)

            unique_top_words.append(word_t)


    top_words = unique_top_words


    with open(output_file, "w") as f:
        for word_t in top_words:
            f.write(word_t[0] + "," + str(word_t[1]) +"\n")

    print("Done.")


authors_query = [('Chengxiang Zhai', 'University of Illinois at Urbana Champaign'), ('Kevin Chenchuan Chang', 'University of Illinois at Urbana Champaign'), ('Jiawei Han', 'University of Illinois at Urbana Champaign')]


authors_query = authors_query[0:1]
for author_t in authors_query:
    profile_researcher(*author_t)
