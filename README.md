Andrew Tsui 56869011
Alvin Truong 43791102
Ashley Teves 10177429
Tiffany Liang 51044452

Installing Dependencies

Make sure all libraries/modules are installed before running the program.
Most notable are networkx, lxml, bs4, flask, nltk, numpy, and simhash.
You can install them on your machine with the command:
"pip3 install --user *library name*"


Building the Index

After installing the libraries, you should be able to run the programs.
If you don't already have the final index which can be found in the partial_indexes directory,
then you will need to run the inverted index builder.
You can use this command in the root directory of the project:
"python3 inverted_index.py"


Running the Search Engine

If you already have the final index in the partial_indexes directory,
then there is no need to run the inverted index builder.
Instead, you can use this command to run the CLI GUI:
"python3 search_engine.py"

If you would like to use the web GUI, you can use the command:
"python3 gui.py"
This web GUI is built using the flask web application framework.
After running the command, a browser should open automatically in your default browser.
If not, just go to 127.0.0.1:5000 in your browser to access our web GUI.