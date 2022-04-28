import json

papers_file = '../../testAbstracts.json'

with open(papers_file, 'r') as f:
    papers = json.load(f)
    papers[0]['foo'] = 'bar'
    print(papers[0])


with open(papers_file, 'w') as f:
    json.dump(papers[0], f)

    # json.dump({'data': papers}, f)
