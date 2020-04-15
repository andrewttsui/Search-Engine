import json, re, os, sys, time, math
from bs4 import BeautifulSoup
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
import numpy as np
from collections import defaultdict


# ***** START OF BOOLEAN RETRIEVAL ***** #

def find_documents(query, index_offset, pos_offset, titles, snippets, page_rank):
    matched_doc = set()
    ss = SnowballStemmer('english')
    inverted_index = open('partial_indexes/final_index.txt', 'r')
    pos_index = open('partial_indexes/final_pos.txt', 'r')
    docs = open('partial_indexes/docs.txt', 'r')
    num_lines = int(docs.readline())
    url_dict = json.load(docs)
    docs.close()

    N = 0
    word_count = defaultdict(int)
    word_indexes = defaultdict(list)
    scores = defaultdict(float)
    length = defaultdict(float)
    matched_docs = set()
    location = dict()
    result = list()

    words = {ss.stem(word.lower()) for word in query.split() if word not in stopwords.words('english')}

    if len(words) == 1:  # if the query is only 1 word long, don't do cosine
        word = list(words)[0]
        if word in index_offset.keys():
            word_entry = get_index_entry(word, inverted_index, index_offset)
            for doc, score in sorted(word_entry.items(), key=lambda x: (-x[1]))[:10]:
                result.append((url_dict[doc], titles[doc], snippets[doc]))
            inverted_index.close()
            pos_index.close()
            return result
        else:  # return empty set if no term matches in index
            inverted_index.close()
            pos_index.close()
            return {}
    else:  # if the query is more than 1 word long, do cosine
        for word in words:
            if word in index_offset.keys():
                word_entry = get_index_entry(word, inverted_index, index_offset)
                word_count[word] += 1  # count occurrences or word in query
                word_indexes[word] = word_entry  # keep track of postings list

                if not matched_docs:
                    matched_docs = set(word_entry.keys())
                else:
                    matched_docs.intersection_update(set(word_entry.keys()))

            else:  # if no term matches in index, continue
                continue

        for v in word_indexes.values():
            N += len(v)

        for word, count in word_count.items():
            word_entry = word_indexes[word]
            weight_tq = (1 + math.log10(count)) * (math.log10(N / float(len(word_entry))))  # calculate tf-idf for query words
            for doc, score in word_entry.items():
                if doc in matched_doc: # take position of word into account if in shared docs
                    pos_entry = get_pos_entry(word, pos_index, pos_offset) 
                    location[doc] = pos_entry[doc]

                # calculate score for doc
                scores[doc] += (score * weight_tq)
                length[doc] += score ** 2
                if doc in matched_doc:
                    scores[doc] += location[doc]

        for doc in scores:
            if doc in page_rank:
                scores[doc] = (scores[doc] / math.sqrt(length[doc]),page_rank[doc])
            else:
                scores[doc] = (scores[doc] / math.sqrt(length[doc]),float(0))


        for doc, score in sorted(scores.items(), key=lambda x: (-x[1][0],-x[1][1]))[:10]:
            result.append((url_dict[doc], titles[doc], snippets[doc]))
        inverted_index.close()
        pos_index.close()
        return result


# ***** END OF BOOLEAN RETRIEVAL ***** #

def get_index_entry(word, f, line_offset):
    f.seek(line_offset[word], 0)
    # get line from final index
    line = f.readline().strip()
    split_line = line.split(":")
    # create dictionary from inner dictionary
    return json.loads(':'.join(split_line[1:])[:-1])

def get_pos_entry(word, f, line_offset):
    f.seek(line_offset[word], 0)
    # get line from final index
    line = f.readline().strip()
    split_line = line.split(":")
    # create dictionary from inner dictionary
    return json.loads(':'.join(split_line[1:])[:-2])

def load_index_offset():
    offset_file = open('partial_indexes/index_offset.txt', 'r')
    line_offset = json.load(offset_file)
    offset_file.close()
    return line_offset

def load_pos_offset():
    offset_file = open('partial_indexes/pos_offset.txt', 'r')
    line_offset = json.load(offset_file)
    offset_file.close()
    return line_offset

def load_titles():
    title_file = open('partial_indexes/titles.txt', 'r')
    titles = json.load(title_file)
    title_file.close()
    return titles

def load_snippets():
    snippet_file = open('partial_indexes/snippets.txt', 'r')
    snippets = json.load(snippet_file)
    snippet_file.close()
    return snippets

def load_page_rank():
    page_rank_file = open('partial_indexes/ranked_pages.txt', 'r')
    page_rank = json.load(page_rank_file)
    page_rank_file.close()
    return page_rank


if __name__ == "__main__":
    index_offset = load_index_offset()
    pos_offset = load_pos_offset()
    titles = load_titles()
    snippets = load_snippets()
    page_rank = load_page_rank()
    search = input("Search: ")
    while search != 'q':
        current_time = time.time()
        result = find_documents(search, index_offset, pos_offset, titles, snippets, page_rank)
        if len(result) == 0:
            print('No results')
        for link in result:
            print(link)
        print(time.time() - current_time)
        search = input("Search: ")