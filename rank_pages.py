import networkx as nx 
import json

def dict_to_diGraph(d):
    diG = nx.DiGraph()
    for k, v in d.items():
        for i in v:
            diG.add_edge(k, i)
    return diG

def create_pagerank():
    # initialize variables
    outgoing_links = dict()
    
    # load outgoingLinks.txt file
    outgoing_file = open('partial_indexes/outgoing.txt','r')
    outgoing_links = json.load(outgoing_file)
    outgoing_file.close()
    
    # create directed graph
    diG = dict_to_diGraph(outgoing_links)
    
    # calculate page rank
    pr = nx.pagerank(diG, max_iter = 25)
    # creates a dict of doc_id: page rank
    
    # write page rank to file
    with open('partial_indexes/ranked_pages.txt', 'w') as out:
        json.dump(pr, out)
    