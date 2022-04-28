import launch
import json
import random

outfile = "../../data/InspecRes.json"

papers_root = '../../data/Inspec/docsutf8/'
keywords_root = '../../data/Inspec/keys/'

embedding_distributor = launch.load_local_embedding_distributor()
pos_tagger = launch.load_local_corenlp_pos_tagger()


experiment_json = []
total_papers = 2000
num_test = 50

test_paper_idxs = random.sample(range(2, total_papers), num_test)

for paper_idx in test_paper_idxs:
    paper_file = papers_root + str(paper_idx) + ".txt"
    keywords_file = keywords_root + str(paper_idx) + ".key"

    try:
        with open(paper_file) as p_f, open(keywords_file) as k_f:
            paper_json = {}
            raw_text = p_f.read()
            labeled_keywords_p = k_f.read().split("\n")
            paper_json['abstract'] = raw_text

            predicted_keywords_p = launch.extract_keyphrases(embedding_distributor, pos_tagger, raw_text, 10, 'en')

            paper_json['keywords_actual'] = labeled_keywords_p
            paper_json['keywords_predicted'] = predicted_keywords_p
            experiment_json.append(paper_json)
    except:
        print("Skipping missing file... ")


with open(outfile, "w") as f:
    json.dump(experiment_json, f, indent=4)
