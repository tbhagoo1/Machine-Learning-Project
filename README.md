# Running NMF Model:

1. Activate virtual env prior to executing code.
2. Execute `pip install -r requirements.txt` from the main project directory to install necessary packages. A Linux environment may be needed if C++ build tools are not installed in Windows. Wordcloud installation may also fail on Windows without installation file.
3. In `nmf.py` set global variables for dataset and topic count. The default values are: `DATASET_TYPE = "food"` `NUM_TOPICS = 9`. The `DATASET_TYPE` may be changed to `reviews` if using product review dataset.
4. Execute `python3 nmf.py`. Note this may take 2-3 minutes to process the food dataset!