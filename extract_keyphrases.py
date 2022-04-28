import launch
import json

papers_file = '../../data/Papers/embedrank-customSetNoDiv.json'

embedding_distributor = launch.load_local_embedding_distributor()
pos_tagger = launch.load_local_corenlp_pos_tagger()


paper_keywords = []
with open(papers_file) as f:
    papers = json.load(f)
    # paper_idx = 0

    for paper_idx in range(len(papers)):
        paper_title = papers[paper_idx]['title']
        raw_text = papers[paper_idx]['abstract']

        keywords_p = launch.extract_keyphrases(embedding_distributor, pos_tagger, raw_text, 10, 'en', beta=1)
        print(keywords_p)
        # papers[paper_idx]['keywords'] = keywords_p
        # paper_keywords.append(keywords_p)


# with open(papers_file, "w") as f:
#     json.dump(papers, f, indent=4)
