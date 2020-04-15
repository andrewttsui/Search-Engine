from flask import Flask, render_template, request
import search_engine

index_offset = dict()
pos_offset = dict()
titles = dict()
snippets = dict()
page_rank = dict()

app = Flask(__name__)

result = []

@app.route('/')
def search():
    return render_template('layout.html')


@app.route('/', methods=['POST', 'GET'])
def my_form_post():
    query = request.form.get('query')

    result = search_engine.find_documents(query, index_offset, pos_offset, titles, snippets, page_rank)

    if query == None or query == "":
        return render_template('empty_query.html')


    if len(result) == 0:
        return render_template('empty_results.html')

    return render_template('search.html', data=result)


if __name__ == '__main__':
    index_offset = search_engine.load_index_offset()
    pos_offset = search_engine.load_pos_offset()
    titles = search_engine.load_titles()
    snippets = search_engine.load_snippets()
    page_rank = search_engine.load_page_rank()
    app.run(debug=True, use_reloader=False)