import pandas as pd
import requests
from bs4 import BeautifulSoup
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from nltk.stem import WordNetLemmatizer
from sklearn.cluster import KMeans
import nltk
from ast import literal_eval
import streamlit as st
import spacy
from scipy.spatial.distance import cosine
import numpy as np

nltk.download('wordnet') 
nltk.download('stopwords')


st.title('Research Buddy ...')
st.markdown('<style>body{background-color:#F0F2F6;}h1{text-align:center;color:#333333;}h2{color:#666666;}input[type="text"]{border-radius:5px;padding:8px;font-size:20px;}button{background-color:#007BFF;color:#FFFFFF;padding:12px;border:none;border-radius:5px;font-size:16px;cursor:pointer;margin-top:10px;}button:hover{background-color:#0056b3;}</style>', unsafe_allow_html=True)
st.markdown('## Search for Principal Investigators')
location = st.selectbox("Select the Location", ("IITGN", "IITGOA","IITBHU","IITB","ALL IITs"))

df=pd.DataFrame()

if location == "IITGN":
    df = pd.read_csv('IIT_Gandhinagar.csv',converters={'Research Interests':literal_eval})
elif location=='IITGOA':
    df = pd.read_csv('IITGoa_Faculty.csv',converters={'Research Interests':literal_eval}) 

elif location=="IITBHU":
    df=pd.read_csv('IIT_BHU.csv',converters={'Research Interests':literal_eval})
elif location =="IITB":
    df=pd.read_csv('IITB_CSE.csv',converters={'Research Interests':literal_eval})
else:
    df = pd.read_csv('Combined_Faculty.csv',converters={'Research Interests':literal_eval})




NUM_CLUSTERS = 10

def preprocess(text):
    
    text = text.lower()
    words = text.split()
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    words = sorted(list(set(words)))
    text = ' '.join(words)

    return text



def find_synonyms(query):
  synonyms = set()
  for word in query.split():
      for syn in wordnet.synsets(word):
          for lemma in syn.lemmas():
              synonyms.add(lemma.name().replace('_', ' '))
  return list(synonyms)

def get_scholar_data(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36 Edge/16.16299'
    }
    res = requests.get(url, headers=headers)
    soup = BeautifulSoup(res.text, 'html.parser')
    publications = []

    def extract_publication(pub):
        try:
            title = pub.select('.gsc_a_at')[0].text.strip()
            venue = pub.select('.gsc_a_jc')[0].text.strip()
            year = int(pub.select('.gsc_a_y')[0].text.strip())
            if year >= (pd.datetime.now().year - 5):
                return (title, venue, year)
        except:
            pass

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(extract_publication, pub) for pub in soup.select('#gsc_a_b .gsc_a_tr')]

        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result is not None:
                publications.append(result)

    return publications

vectorizer = TfidfVectorizer(preprocessor=preprocess)
vectorizer.fit(df['Research Interests'].apply(lambda x: ' '.join(x)))

reg = LinearRegression()
X = vectorizer.transform(df['Research Interests'].apply(lambda x: ' '.join(x)))
y = df['h_index']
reg.fit(X, y)

km = KMeans(n_clusters=NUM_CLUSTERS, random_state=42)
km.fit(X)

import concurrent.futures

def search(query):
    
    
  
    query = preprocess(query)
    query_synonyms = find_synonyms(query)
    query_vec = vectorizer.transform([query] + query_synonyms)
    cluster = km.predict(query_vec)[0]
    

    df_cluster = df[km.labels_ == cluster]
    
    relevance = reg.predict(query_vec)[:]
    ranked=[]

    exact_match = False
    for idx, row in df.iterrows():
        professor_name = row['Faculty Name'].lower()
        if query in professor_name:
            exact_match = True
            ranked.append((row['Faculty Name'], row['Institution'],row['HomePage'], row['Scholar Url'], row['Image']))
            
    
    if exact_match:
      return ranked
    

    for idx, row in df.iterrows():
        interests = row['Research Interests']
        for interest in interests:
            if (query in interest.lower()):
                ranked.append((row['Faculty Name'], row['Institution'],row['HomePage'], row['Scholar Url'], row['Image']))
                break
    
    
    nlp = spacy.load("en_core_web_md")
    
        

    def process_row(row):
        if row['h_index'] == float('nan') or pd.isna(row['h_index']):
            return None
        research_interests = row['Research Interests']
        embeddings = []
        for interest in research_interests:
            embeddings.append(nlp(interest).vector)
        
        similarity = 1 - cosine(nlp(query).vector, np.mean(embeddings,axis=0))
        
        if similarity<0.69:
            return None

        h_index = row['h_index']
        i10_index = row['i10_index'] if not pd.isna(row['i10_index']) else 0
        scholar_url = row['Scholar Url']
        if pd.isna(scholar_url):
            return None

        try:
            publications = get_scholar_data(scholar_url)
        except:
            publications = []

        recent_activity = sum([int(pub[2] >= (pd.datetime.now().year - 5)) for pub in publications])
        rank = 0.5 * (h_index + i10_index) + 0.3 * relevance + 0.2 * recent_activity # weights based on importance
        return (row['Faculty Name'], row['Institution'], row['HomePage'], row['Scholar Url'], row['Image'], rank)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        # futures = [executor.submit(process_row, row) for idx, row in df_cluster.iterrows()]
        futures = [executor.submit(process_row, row) for idx, row in df_cluster.iterrows()]
        results = [future.result() for future in concurrent.futures.as_completed(futures) if future.result() is not None]
    

    results.sort(key=lambda x: max(x[5]), reverse=True)
    ranked.extend(results)
    for i in range(len(ranked)):
        for j in range(i+1,len(ranked)):
            
            if j<len(ranked) and ranked[i][0]==ranked[j][0]:
                ranked.pop(j)

    
    return ranked




query = st.text_input('Enter your query:')
button = st.button('Search')


if button:
    results = search(query)
    if len(results) > 0:
        
        for result in results:
            image_url = result[4]
            image_style = """
                width: 150px;
                height: 100px;
                border-radius: 5px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                """
            st.markdown(f'<img src="{image_url}" style="{image_style}" />', unsafe_allow_html=True)

            faculty_info_style = """
            color: #333333;
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 5px;
            """
            st.markdown(f'<div style="{faculty_info_style}">Faculty Name: {result[0]}</div>', unsafe_allow_html=True)

            institution_style = """
            color: #666666;
            font-size: 16px;
            margin-bottom: 5px;
            """
            st.markdown(f'<div style="{institution_style}">Institution: {result[1]}</div>', unsafe_allow_html=True)

            st.write(f'Homepage: {result[2]}')
            st.write(f'Scholar URL: {result[3]}')
            st.write('\n')
    else:
        st.markdown('No results found.')
