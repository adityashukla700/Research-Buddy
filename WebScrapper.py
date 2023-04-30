import requests
from bs4 import BeautifulSoup
from csv import writer
url='https://iitgn.ac.in/faculty'

r=requests.get(url)
soup=BeautifulSoup(r.content,'lxml')
cards=soup.find_all('div',class_='card__body')
with open('iitgn_faculty2.csv','w',newline='',encoding='utf-8') as f:
    thewriter=writer(f)
    header=['Faculty Name','Field','HomePage','Reasearch Interests','Image']
    thewriter.writerow(header)

    result=[]

    for card in cards:
        result.append(card)

    for card in result:
        prof_name=card.find('a',class_='h5').text
        # print(prof_name)

        field=card.find('strong').text.split('\n')[4].strip()
        # print(field)


        prof_link=card.find('a',class_='h5').get('href')
        # print(prof_link)

        

        prof_page=requests.get(prof_link)
        rsoup=BeautifulSoup(prof_page.content,'lxml')
        research=rsoup.find('div',class_='sidebar__widget')
        parent=research.find('ul')
        if(parent==None):
            break
        all_interest=parent.find_all('li')
        research_interest=[]
        for li in all_interest:
            research_interest.append(li.text)
        # print(research_interest)
        prof_image=rsoup.find('img',class_='border--round').get('src')
        info=[prof_name,field,prof_link,research_interest,prof_image]
        thewriter.writerow(info)
        # print()