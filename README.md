# Running NMF Model:

## NOTE: To execute properly, the `food.csv` and the `reviews.csv` must be located within the archive directory! Ensure both datasets are contained within this directory prior to executing any code.

1. Activate virtual env prior to executing code.
2. Execute `pip install -r requirements.txt` from the main project directory to install necessary packages. A Linux environment may be needed if C++ build tools are not installed in Windows. Wordcloud installation may also fail on Windows without installation file. If wordcloud pip installation fails on Windows, install using the provided .whl file using `pip install <filename.whl>`
3. In `nmf.py` set global variables for dataset and topic count. The default values are: `DATASET_TYPE = "food"` `NUM_TOPICS = 9`. The `DATASET_TYPE` may be changed to `reviews` if using product review dataset.
4. Execute `python3 nmf.py`. Note this may take 2-3 minutes to process the food dataset!