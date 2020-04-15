import json 

def urls_to_ids(outgoing):
    '''convert all keys and values that are urls to doc ids'''
    outgoing_ids = dict()

    # load url_docID.txt
    doc_file = open('partial_indexes/inverted_docs.txt','r')
    docs = json.load(doc_file)
    doc_file.close()

    for page, outgoingLinks in outgoing.items():
        link_ids = [docs[link] for link in outgoingLinks if link in docs]
        # add new entry to dict
        outgoing_ids[page] = link_ids
    
    # write links dict to file
    return outgoing_ids   