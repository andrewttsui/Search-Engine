import json, re, os, sys, time, math, hashlib
from bs4 import BeautifulSoup, Comment
from nltk.stem import SnowballStemmer
from collections import defaultdict
from urllib.parse import urlparse, urldefrag, urljoin
from extract_links import urls_to_ids
from rank_pages import create_pagerank, dict_to_diGraph
from simhash import Simhash, SimhashIndex

inverted_index = dict()
documents = dict()
inverted_docs = dict()
hash_ids = defaultdict(int)
line_offset = dict()
sim_index = SimhashIndex(list(), f=64,k = 5)
exact_dup = set()
titles = dict()
snippets = dict()
position = dict()
outgoing = dict()

def partial_index():
    doc_id = 1
    partial_id = 1
    merge_id = 1
    num_dups = 0
    directory = 'DEV/'
    for subdir in os.listdir(directory):
        for json_file in os.listdir(directory + subdir):
            json_file = directory + subdir + '/' + json_file
            with open(json_file, 'r') as myfile:
                data = json.load(myfile)
                if urlparse(data['url']).fragment != '':
                    continue
                if 'ical' in urlparse(data['url']).query:
                    continue
                if 'eppstein/pix' in data['url']:
                    continue
                
                soup = BeautifulSoup(data['content'], 'lxml')

                checksum = hashlib.md5(soup.get_text().encode('utf-8')).hexdigest()
                if checksum in exact_dup:
                    num_dups += 1
                    continue
                else:
                    exact_dup.add(checksum)

                simhash = Simhash(soup.get_text())
                if len(sim_index.get_near_dups(simhash)) < 5:
                    sim_index.add(data['url'], simhash)
                else:
                    num_dups += 1
                    continue
                print(data['url'])

                documents[doc_id] = data['url']
                inverted_docs[data['url']] = doc_id
                

                for script in soup(["script", "style"]):
                    script.extract()
                for element in soup(text=lambda text: isinstance(text, Comment)):
                    element.extract()
                for img in soup.find_all('img'):
                    img.extract()
                for p in soup.find_all("p", {'class': 'entry-meta'}):
                    p.decompose()
                for p in soup.find_all("p", {'class': 'entry-time'}):
                    p.decompose()

                if soup.title != None:
                    titles[doc_id] = soup.title.string
                else:
                    titles[doc_id] = data['url']

                # EXTRACT LINKS FOR PAGE RANK
                links = list()
                for a in soup.findAll('a'):
                    link = a.get('href')
                    parsed = urlparse(link)
                    # if there is a fragment attached to the url
                    # only add the defragged url
                    if len(parsed.fragment) > 0:
                        link = urldefrag(link)[0]
                    if not bool(parsed.netloc):
                        link = str(urljoin(data['url'], link))
                    if link not in links:
                        links.append(link)
                # add links to dict entry
                outgoing[doc_id] = links

                content = soup.get_text().lower()
                pattern_split = re.compile(r"[^a-z0-9'’]|\n")
                content_list = re.split(pattern_split, content)
                content_list = list(filter(None, content_list))

                token_weights(soup.find_all('a', href=True), doc_id, 'a', content_list)
                token_weights(soup.find_all('title'), doc_id, 't', content_list)
                token_weights(soup.find_all('h1'), doc_id, 'h', content_list)
                token_weights(soup.find_all('h2'), doc_id, 'h', content_list)
                token_weights(soup.find_all('h3'), doc_id, 'h', content_list)
                token_weights(soup.find_all('h4'), doc_id, 'h', content_list)
                token_weights(soup.find_all('bold'), doc_id, 'b', content_list)
                token_weights(soup.find_all('strong'), doc_id, 'b', content_list)
                token_weights(soup.find_all('em'), doc_id, 'b', content_list)
                snippets[doc_id] = ' '.join(soup.get_text().strip().split())[:100]
                tokenize(soup.get_text().lower(), doc_id, 'n', content_list)

                # when the inverted index gets larger than 10.0 Mb, write to file and start over 280000 10000000
                if (sys.getsizeof(inverted_index) > 5000000):
                    partial_index = os.path.join('partial_indexes', 'partial_index' + str(partial_id) + ".txt")
                    pos_index = os.path.join('partial_indexes', 'partial_pos' + str(partial_id) + ".txt")
                    with open(partial_index, 'w') as out:
                        json.dump(inverted_index, out, sort_keys=True, indent=4)
                    with open(pos_index, 'w') as out:
                        json.dump(position, out, sort_keys=True, indent=4)
                    if partial_id == 2:
                        partial_merge_index_files('partial_indexes/partial_index' + str(partial_id - 1) + '.txt', partial_index, 'partial_indexes/merge_index' + str(merge_id) + '.txt')
                        partial_merge_pos_files('partial_indexes/partial_pos' + str(partial_id - 1) + '.txt', pos_index, 'partial_indexes/merge_pos' + str(merge_id) + '.txt')
                        merge_id += 1
                    elif partial_id > 2:
                        partial_merge_index_files('partial_indexes/merge_index' + str(merge_id - 1) + '.txt', partial_index, 'partial_indexes/merge_index' + str(merge_id) + '.txt')
                        partial_merge_pos_files('partial_indexes/merge_pos' + str(partial_id - 1) + '.txt', pos_index, 'partial_indexes/merge_pos' + str(merge_id) + '.txt')
                        merge_id += 1
                    partial_id += 1
                    inverted_index.clear()

            doc_id += 1

    # write the rest of the inverted index to file
    partial_index = os.path.join('partial_indexes', 'partial_index' + str(partial_id) + '.txt')
    pos_index = os.path.join('partial_indexes', 'partial_pos' + str(partial_id) + ".txt")
    with open(partial_index, 'w') as out:
        json.dump(inverted_index, out, sort_keys=True, indent=4)
    with open(pos_index, 'w') as out:
        json.dump(position, out, sort_keys=True, indent=4)

    num_lines = final_merge_index_files('partial_indexes/merge_index' + str(merge_id - 1) + '.txt', partial_index, 'partial_indexes/final_index.txt')
    final_merge_pos_files('partial_indexes/merge_pos' + str(merge_id - 1) + '.txt', pos_index, 'partial_indexes/final_pos.txt')

    # write the doc_ids to file
    with open('partial_indexes/docs.txt', 'w') as out:
        out.write(str(num_lines) + '\n')
        json.dump(documents, out)
    
    with open('partial_indexes/inverted_docs.txt', 'w') as out:
        json.dump(inverted_docs, out)

    outgoing_ids = urls_to_ids(outgoing)
    # remove invalid links (not in corpus)
    with open('partial_indexes/outgoing.txt', 'w') as out:
        json.dump(outgoing_ids, out)

    with open('partial_indexes/titles.txt', 'w') as out:
        json.dump(titles, out)

    with open('partial_indexes/snippets.txt', 'w') as out:
        json.dump(snippets, out)

    # write the line offsets to file
    offset = 0
    with open('partial_indexes/final_index.txt', 'r') as f:
        for line in f.readlines():
            term = line.split(":")[0].strip('"')
            line_offset[term] = offset
            offset += len(line)
    with open('partial_indexes/index_offset.txt', 'w') as out:
        json.dump(line_offset, out)

    offset = 0
    with open('partial_indexes/final_pos.txt', 'r') as f:
        for line in f.readlines():
            term = line.split(":")[0].strip('"')
            line_offset[term] = offset
            offset += len(line)
    with open('partial_indexes/pos_offset.txt', 'w') as out:
        json.dump(line_offset, out)

    ### PageRank ###
    create_pagerank()

    # remove helper files
    partial_dir = "partial_indexes/"
    for item in os.listdir(partial_dir):
        if item.startswith("partial_pos") or item.startswith("merge"):
            os.remove(os.path.join(partial_dir, item))
    print(num_dups)

# will be used to implement the different weights of tokens
def token_weights(tag_list, doc_id, tag_type, content):
    for tag in tag_list:
        tokenize(tag.text.lower(), doc_id, tag_type, content)
        tag.decompose()


def tokenize(text, doc_id, tag, content_list):
    ss = SnowballStemmer('english')
    pattern_split = re.compile(r"[^a-z0-9'’]|\n")
    text_list = re.split(pattern_split, text)
    text_list = list(filter(None, text_list))
    single_char = re.compile(r"(\'|\’)+")

    time_end = time.time()
    for word in text_list:
        # checks the index in content list
        indexes = list(i for i, w in enumerate(content_list) if w == word)
        # if word not in index check for substring
        if not indexes:
            indexes = list(i for i, w in enumerate(content_list) if word in w)
            # if word not in index again give default of -1 for position
            if not indexes:
                indexes = [-1]
        stemmed = ss.stem(word)
        if time.time() - time_end > 5:
            break
        if re.match(single_char, word):
            continue
        # write in position for term in doc_id
        if stemmed not in position:
            position[stemmed] = {doc_id: indexes}
        else:
            if doc_id not in position:
                position[stemmed][doc_id] = indexes
            else:
                position[stemmed][doc_id].extend(indexes)

        if stemmed not in inverted_index:
            inverted_index[stemmed] = {doc_id: [1, {'a': 0, 't': 0, 'h': 0, 'b': 0}]}
        else:
            if doc_id not in inverted_index[stemmed]:
                inverted_index[stemmed][doc_id] = [1, {'a': 0, 't': 0, 'h': 0, 'b': 0}]
            else:
                inverted_index[stemmed][doc_id][0] += 1
        if tag == 'n':
            continue
        inverted_index[stemmed][doc_id][1][tag] += 1

### ---- START OF MERGING ---- ###

def partial_merge_index_files(initial_file, next_file, output_file):
    print('MERGING', initial_file, 'WITH', next_file, 'INTO', output_file)
    out = open(output_file, "w")
    initial_file = open(initial_file, 'r')
    next_file = open(next_file, 'r')
    initial_file.readline()
    next_file.readline()
    out.write('{\n')

    initial_dict = retrieve_term(initial_file)
    next_dict = retrieve_term(next_file)

    first = True
    while initial_dict != None or next_dict != None:

        if initial_dict == None:
            string = json.dumps(next_dict)
            next_dict = retrieve_term(next_file)
        elif next_dict == None:
            string = json.dumps(initial_dict)
            initial_dict = retrieve_term(initial_file)
        else:
            initial_term = next(iter(initial_dict))
            next_term = next(iter(next_dict))
            if initial_term == next_term:
                initial_dict[initial_term].update(next_dict[next_term])
                string = json.dumps(initial_dict)
                initial_dict = retrieve_term(initial_file)
                next_dict = retrieve_term(next_file)
            elif initial_term < next_term:
                string = json.dumps(initial_dict)
                initial_dict = retrieve_term(initial_file)
            else:
                string = json.dumps(next_dict)
                next_dict = retrieve_term(next_file)
        entry = string.split()
        if first:
            first = False
            out.write(entry.pop(0)[1:] + '{\n')
        else:
            out.write(',\n' + entry.pop(0)[1:] + '{\n')
        for key, freq, t1, t2, t3, t4, t5, t6, t7, t8 in zip(entry[0::10], entry[1::10], entry[2::10], entry[3::10],\
            entry[4::10], entry[5::10], entry[6::10], entry[7::10], entry[8::10], entry[9::10]):
            tag = t1 + t2 + t3 + t4 + t5 + t6 + t7 + t8
            if key[0] == '{':
                if tag[-1] == '}':
                    string = key[1:] + ' ' + freq + ' ' + tag[:-2] + '\n}'
                else:
                    string = key[1:] + ' ' + freq + ' ' + tag + '\n'
            elif tag[-1] == '}':
                string = key + ' ' + freq + ' ' + tag[:-2] + '\n}'
            else:
                string = key + ' ' + freq + ' ' + tag + '\n'
            out.write(string)
    out.write('\n}')
    initial_file.close()
    next_file.close()
    out.close()

def partial_merge_pos_files(initial_file, next_file, output_file):
    print('MERGING', initial_file, 'WITH', next_file, 'INTO', output_file)
    out = open(output_file, "w")
    initial_file = open(initial_file, 'r')
    next_file = open(next_file, 'r')
    initial_file.readline()
    next_file.readline()
    out.write('{\n')

    initial_dict = retrieve_term(initial_file)
    next_dict = retrieve_term(next_file)

    first = True
    while initial_dict != None or next_dict != None:

        if initial_dict == None:
            string = json.dumps(next_dict)
            next_dict = retrieve_term(next_file)
        elif next_dict == None:
            string = json.dumps(initial_dict)
            initial_dict = retrieve_term(initial_file)
        else:
            initial_term = next(iter(initial_dict))
            next_term = next(iter(next_dict))
            if initial_term == next_term:
                initial_dict[initial_term].update(next_dict[next_term])
                string = json.dumps(initial_dict)
                initial_dict = retrieve_term(initial_file)
                next_dict = retrieve_term(next_file)
            elif initial_term < next_term:
                string = json.dumps(initial_dict)
                initial_dict = retrieve_term(initial_file)
            else:
                string = json.dumps(next_dict)
                next_dict = retrieve_term(next_file)
        entry = string.split()
        if first:
            first = False
            out.write(entry.pop(0)[1:] + '\n')
        else:
            out.write(',\n' + entry.pop(0)[1:] + '\n')
        out.write(' '.join(entry)[:-1])
    out.write('\n}')
    initial_file.close()
    next_file.close()
    out.close()

def final_merge_pos_files(initial_file, next_file, output_file):
    print('MERGING', initial_file, 'WITH', next_file, 'INTO', output_file)
    out = open(output_file, "w")
    initial_file = open(initial_file, 'r')
    next_file = open(next_file, 'r')
    initial_file.readline()
    next_file.readline()

    initial_dict = retrieve_term(initial_file)
    next_dict = retrieve_term(next_file)

    first = True
    while initial_dict != None or next_dict != None:

        if initial_dict == None:
            string = json.dumps(next_dict)
            next_dict = retrieve_term(next_file)
        elif next_dict == None:
            string = json.dumps(initial_dict)
            initial_dict = retrieve_term(initial_file)
        else:
            initial_term = next(iter(initial_dict))
            next_term = next(iter(next_dict))
            if initial_term == next_term:
                initial_dict[initial_term].update(next_dict[next_term])
                string = json.dumps(initial_dict)
                initial_dict = retrieve_term(initial_file)
                next_dict = retrieve_term(next_file)
            elif initial_term < next_term:
                string = json.dumps(initial_dict)
                initial_dict = retrieve_term(initial_file)
            else:
                string = json.dumps(next_dict)
                next_dict = retrieve_term(next_file)
        if first:
            first = False
            out.write(string[1:-1])
        else:
            out.write(",\n" + string[1:])
    initial_file.close()
    next_file.close()
    out.close()


def final_merge_index_files(initial_file, next_file, output_file):
    print('MERGING', initial_file, 'WITH', next_file, 'INTO', output_file)
    num_lines = 0
    out = open(output_file, "w")
    initial_file = open(initial_file, 'r')
    next_file = open(next_file, 'r')
    initial_file.readline()
    next_file.readline()

    initial_dict = retrieve_term(initial_file)
    next_dict = retrieve_term(next_file)

    first = True
    while initial_dict != None or next_dict != None:
        write_dict = dict()
        if initial_dict == None:
            string = calculate_tfidf(next_dict)
            next_dict = retrieve_term(next_file)
        elif next_dict == None:
            string = calculate_tfidf(initial_dict)
            initial_dict = retrieve_term(initial_file)
        else:
            initial_term = next(iter(initial_dict))
            next_term = next(iter(next_dict))
            if initial_term == next_term:
                initial_dict[initial_term].update(next_dict[next_term])
                string = calculate_tfidf(initial_dict)
                initial_dict = retrieve_term(initial_file)
                next_dict = retrieve_term(next_file)
            elif initial_term < next_term:
                string = calculate_tfidf(initial_dict)
                initial_dict = retrieve_term(initial_file)
            else:
                string = calculate_tfidf(next_dict)
                next_dict = retrieve_term(next_file)

        if first:
            out.write(string[1:-1])
            first = False
        else:
            out.write(",\n" + string[1:-2] + "}")
        num_lines += 1

    initial_file.close()
    next_file.close()
    out.close()
    return num_lines


def retrieve_term(f):
    dict_object = ''
    d = None
    current_time = time.time()
    while True:
        if time.time() - current_time > 5:
            return None
        line = f.readline().strip()
        dict_object = dict_object + line
        if line == '},':
            if dict_object[0] == '{':
                dict_object = dict_object[1:]
            dict_object = '{' + dict_object[:-1] + '}'
            d = json.loads(dict_object)
            break
    return d


def calculate_tfidf(d):
    N = len(documents)
    first = True
    square_sum_tfidf = 0
    write_dict = dict()
    for key, value_to_list in d.items():
        word = key
        docs = dict(value_to_list)
    for doc, posting in docs.items():
        freq = posting[0]
        tags = posting[1]
        tfidf = (1 + math.log10(int(freq))) * (math.log10(float(N) / len(d.items())))
        weights = (tags['a'] * 0.5) + (tags['b'] * 0.75) + (tags['h'] * 1.5) + (tags['t'] * 2)
        if word not in write_dict:
            write_dict[word] = {doc: tfidf + weights}
        else:
            write_dict[word][doc] = tfidf + weights
    sort_dict = sorted(write_dict[word].items(), key=lambda kv: (-kv[1]))
    write_dict.clear()
    for entry in sort_dict:
        if word not in write_dict:
            write_dict[word] = {entry[0]: entry[1]}
        else:
            write_dict[word][entry[0]] = entry[1]
    string = json.dumps(write_dict)
    return string

### ---- END OF MERGING ---- ###

if __name__ == "__main__":
    partial_index()
