import sys
import os
import bs4
import urllib.request, urllib.error, urllib.parse
import http.client
import ssl 
import multiprocessing as mp
from socket import error as SocketError
import MeCab
import plyvel
import pickle
import copy
import re
import concurrent.futures
def html_adhoc_fetcher(url, db):
  html = None
  for _ in range(5):
    opener = urllib.request.build_opener()
    TIME_OUT = 1.0
    try:
      html = opener.open(str(url), timeout = TIME_OUT).read()
    except Exception as e:
      print(e)
      continue
  if html == None:
    return (None, None, None, None)
 
  soup = bs4.BeautifulSoup(html)
  title = (lambda x:str(x.string) if x != None else 'Untitled')( soup.title )
  links =  ['http://ncode.syosetu.com' + x for x in \
              [x for x in [ a['href'] for a in soup.find_all('a',href=True) ] \
                if x[0] == '/' and re.search('/[0-9a-z]{1,}/\d{1,}/', x)]]
    
  links = list(set(links))
  return (html, title,  links, soup)


def stemming_pair(soup):
  contents = soup.findAll('div', {'class': 'novel_view'})
  content  = ''.join( [x.text for x in contents] )
  content  = re.sub('\s{1,}', '\n', content)
  textlist = re.sub('\s{1,}', '\n', content ).split('\n')
  textlistd= copy.copy(textlist) 
  textlist.insert(0, 'None')
  zipped   = list(zip(textlist, textlistd ))
  zipped.pop(0)
  zipped.pop(0)
  zipped.pop()
    
  return content

def iter_get(url):
  seedurl = '{url}1/'.format(url=url)
  html, title, links, soup = html_adhoc_fetcher(seedurl, db) 
  print(title)
  zipped = stemming_pair(soup)
  pairs  = [ (bytes(seedurl, 'utf-8'), bytes(zipped, 'utf-8') ) ]
  linkstack = links
  allready = set( seedurl )
  while linkstack != []:
    link = linkstack.pop()
    if link not in allready:
      html, title, links, soup = html_adhoc_fetcher(link, db) 
      zipped = stemming_pair(soup)
      #db.put(bytes(link, 'utf-8'), bytes(zipped, 'utf-8'))
      pairs.append( (bytes(link, 'utf-8'), bytes(zipped, 'utf-8')) )
      print(str(link))
      allready.add(link)
      linkstack.extend(links)
  return pairs
if __name__ == '__main__':
  db = plyvel.DB('./url_contents_pair.ldb', create_if_missing=True)
  tagger = MeCab.Tagger("-Owakati")
  if '--step1' in sys.argv:
    urls = []
    with open("vars/narou.urls", "r") as f:
      for url in f:
        url = url.strip()
        urls.append(url)

    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
      for pairs in executor.map(iter_get, urls):
        for (key, val) in pairs:
          db.put( key, val )

  
  if '--step2' in sys.argv:
    m = MeCab.Tagger("-Owakati")
    for link, text in db:
      for line in text.decode("utf-8").split("\n"):
        print(m.parse(line).strip())
