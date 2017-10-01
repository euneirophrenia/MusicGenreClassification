import requests
from bs4 import BeautifulSoup as bs
import urllib.request
import urllib.parse
import subprocess
import dryscrape



def isMp3(url):
    return str(url).endswith('.mp3')

def nameAndUrl(url):
    return url.split('/')[-1], 'http://www.amclassical.com/'+url


def getAllLinksFromPageThat(page, func=lambda x: True, outputfunc=lambda x:x):
    res=[]
    r = requests.get(page)
    soup = bs(r.text, "lxml")
    for link in soup.findAll('a'):
        href = link.get('href')
        if func(href):
            res.append(outputfunc(href))

    return res




outuputdest = './MP3/Classical/'


yall = getAllLinksFromPageThat('http://www.amclassical.com/wedding/',
                               lambda x: isMp3(x), lambda x: nameAndUrl(x))



for name, url in yall:
    print(url)
    try:
        rq = urllib.request.Request(url)
        res = urllib.request.urlopen(rq)
        destfile = open(outuputdest + name, 'wb')
        destfile.write(res.read())
        destfile.close()
    except urllib.request.HTTPError as e:
        print("Error processing url",url,'(',e.msg,')')

print("Done downloading ", len(yall))