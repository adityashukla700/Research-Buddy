**The Research Buddy** Search Engine aims to bridge the gap between the students and their potential PIs.

The data provided here is relevant and sufficient to run the project.


The code provided here contains 3 files:

1) ResearchBuddy.py- This the main file that contains the search engine. For running this file streamlit should be installed.

2) WebScrapper.py- This contains the code for scraping data from IITGN faculty profile. Similar techniques have been used to scrape data of other institutes.

3) Preprocessing_Data.ipynb- This contains the code to get more required information after getting data from web scrapping like professor's scholar url, their h and i10 index etc. for all institutes.

-------------------------------------------------

Dependencies- You need to install following dependencies before running the program.

1) BeautifulSoup
2) requests
3) nltk
4) sklearn
5) spacy
6) streamlit.


-------------------------------------------------

Running the Search Engine:

1) Open ResearchBuddy.py and write the following command on the terminal-
streamlit run 	ResearchBuddy.py

2) This will run the file and you will be able to use the search engine.