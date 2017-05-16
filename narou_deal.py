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
import glob
import random
import concurrent.futures
from gensim.models import KeyedVectors
import json
import math
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

  """ なろうを形態素解析 """
  if '--step2' in sys.argv:
    m = MeCab.Tagger("-Owakati")
    for link, text in db:
      for line in text.decode("utf-8").split("\n"):
        print(m.parse(line).strip())
  
  """ Yahoo NewsをN件取得 -> 形態素解析 """
  if '--step3' in sys.argv:
    m = MeCab.Tagger("-Owakati")
    N = 40000
    for gi, name in enumerate( glob.glob("vars/output/*/*") ):
      if gi > N: 
        break
      with open(name) as f:
        for line in f:
          line = line.strip()
          if line == "": 
            continue
          print(m.parse(line).strip())

  """ Yahoo Newとなろうを混ぜる """
  if '--step4' in sys.argv:
    with open("./vars/narou.txt", "r") as fn, open("./vars/yahoo_news.txt", "r") as fy:
      ys = [line.strip().strip() for line in fy]
      ns = [line.strip().strip() for line in fn]
      
      zs = ys + ns
      random.shuffle(zs)
    with open("./vars/zs.txt", "w")  as f:
      for z in zs:
        f.write("%s\n"%z)

  """ 混ぜた結果から、fastText char n gramを停止した状態でベクトル化する 200次元 """
  if '--step5' in sys.argv:
    os.system("./fasttext skipgram -dim 256 -maxn 0 -minCount 1 -input vars/zs.txt -output vars/model")

  """ fasttextモデルをgensimインターフェースで呼べるようにする """
  if '--step6' in sys.argv:
    model = KeyedVectors.load_word2vec_format('vars/model.vec', binary=False)
    open('vars/fasttext.gensim-model.pkl', 'wb').write(pickle.dumps(model) )

    scores = model.most_similar(positive=["眠い"], topn=50) 
    print( json.dumps(scores, ensure_ascii=False, indent=2) )
 
  """ ラベル付をする """
  if '--step7' in sys.argv:
    with open("./vars/narou.txt", "r") as fn, open("./vars/yahoo_news.txt", "r") as fy:
      ys = ["{} {}".format("__label__1", line.strip()) for line in fy]
      ns = ["{} {}".format("__label__2", line.strip()) for line in fn]
      
      zs = ys + ns
      random.shuffle(zs)
    with open("./vars/label_mix.txt", "w")  as f:
      for z in zs:
        f.write("%s\n"%z)
  
  """ テキストのベクトル化を学習する """
  if '--step8' in sys.argv:
    os.system("./fasttext supervised -minCount 1 -dim 256 -epoch 5 -input vars/label_mix.txt -output vars/text")  

  
  """ なろうから変換候補となる単語（名詞  and カタカナ and 漢字）を取り出す """
  if '--step9' in sys.argv:
    m = MeCab.Tagger("-Ochasen")
    term_freq = {}
    for line in open("vars/narou.txt", "r"):
      line = line.strip().replace(" ", "")
      for e in filter( lambda x:"名詞" in x and "一般" in x,\
          m.parse(line).strip().split("\n") ):
        """ 漢字マッチング """
        kan = re.search(u'[\u4E00-\u9FFF]+', \
          e.split("\t")[0], \
          re.UNICODE)
        """ ひらがなのマッチング """
        hira = re.search(u'[\u3040-\u309Fー]+', e.split("\t")[0], re.U)

        """ カタカナのマッチング """
        kana = re.search(u'[\u30A0-\u30FF]+', e.split("\t")[0], re.U)
        
        if kan is not None and hira is None and kana is None:
          term = kan.group(0)
          if term_freq.get(term) is None: 
            term_freq[term] = 0. 
            print(term)
          term_freq[term] += 1.
          ...
        if kan is None and hira is None and kana is not None:
          term = kana.group(0)
          if term_freq.get(term) is None: 
            term_freq[term] = 0. 
            print(term)
          term_freq[term] += 1.
        #print(e)
    open("vars/term_freq.pkl", "wb").write( pickle.dumps(term_freq) )

  """ 作成したモデルから文章のベクトルを作る """
  if '--step10' in sys.argv:
    with open("linear.txt", "w") as f:
      to_writes = []
      for vec in os.popen("./fasttext print-sentence-vectors vars/text.bin < vars/narou.txt").read().strip().split("\n") :
        val = " ".join( [ "%d:%s"%(e+1, n) for e, n in enumerate(vec.split()) ])
        tag = "1"
        to_writes.append("%s %s\n"%(tag, val))
      for vec in os.popen("./fasttext print-sentence-vectors vars/text.bin < vars/yahoo_news.txt").read().strip().split("\n") :
        val = " ".join( [ "%d:%s"%(e+1, n) for e, n in enumerate(vec.split()) ])
        tag = "0"
        to_writes.append("%s %s\n"%(tag, val))
      random.shuffle(to_writes)
      for data in to_writes:
        f.write(data)
 
  """ 文章のベクトルを判別問題としてロシスティック回帰で解く """
  if '--step11' in sys.argv:
    os.system('./train -s 0 linear.txt') 

    os.system('mv linear.txt vars/linear.txt')

  """ 小さいサンプルで、実行する """
  if '--step12' in sys.argv:
    m = MeCab.Tagger("-Owakati")
    sample = m.parse("漫画誌「週刊少年ジャンプ」（集英社）の印刷部数（印刷証明付き）が、今年1～3月の平均で191万5000部となり、200万部を割り込んだことが分かった。(ITmedia ビジネスオンライン)").strip()
    open("sample.txt", "w").write(sample)
    raw = os.popen('echo "{}" | ./fasttext print-sentence-vectors vars/text.bin'.format(sample)).read().strip()
    vec = [float(n) for e,n in enumerate(raw.split())] 
    # 極端な出力になる
    #os.system("./predict -b 0 sample.svm linear.txt.model result")

    with open("./linear.txt.model", "r") as f:
      next(f)
      next(f)
      next(f)
      next(f)
      next(f)
      next(f)
      
      score = 0.
      for w,v in zip(f, vec):
        w = float(w.strip())
        #print(w*v)
        score += w*v

      last = 1. / (1. + math.pow(math.e, -1*score)) 
      print(last)
 
  """ ベースに対してどう変えたらもっともなろうチックなのか探索する """
  if '--step13' in sys.argv:
    m = MeCab.Tagger("-Ochasen")
    term_freq = pickle.loads( open("vars/term_freq.pkl", "rb").read() )
     
    model = pickle.loads( open('vars/fasttext.gensim-model.pkl', 'rb').read() )
    
    ws = []
    with open("./linear.txt.model", "r") as f:
      next(f)
      next(f)
      next(f)
      next(f)
      next(f)
      next(f)
      for w in f:
        w = float(w.strip())
        ws.append(w)

    sample = m.parse("秋篠宮ご夫妻の長女、眞子さまが国際基督教大学（ＩＣＵ）の同級生だった男性と婚約に向けた準備を進めていることが１６日、宮内庁関係者への取材で分かった。眞子さまが秋篠宮ご夫妻に男性を紹介し、交際を認められているという。(産経新聞)").strip()

    
    terms  = list( filter(lambda x:len(x) >= 4, map(lambda x:x.split("\t"), sample.split("\n") ) ) )
    while True:
      sampling = []
      for e, term in enumerate(terms): 
        meta = term[3]
        if '名詞' not in meta:
          continue
        if '数' in meta:
          continue
        #print(term)
        
        try:
          scores = model.most_similar(positive=[term[0]], topn=30)
          scores = list( filter(lambda x:x[0] in term_freq, scores) )
        except KeyError as e:
          continue
       
        #print( scores )
        #continue
        #print( json.dumps(scores, ensure_ascii=False, indent=2) )
        for score in scores:
          deepfurigana, prob = score   
          #print(deepfurigana)
          tmps  = [x[0] for x in terms]
          #print(tmps)
          tmps[e]            = deepfurigana 
          res                = " ".join(tmps)
          #print(res)
          raw = os.popen('echo "{}" | ./fasttext print-sentence-vectors vars/text.bin'.format(res)).read().strip()
          vec = [float(n) for e,n in enumerate(raw.split())] 
          score = 0.
          for w,v in zip(ws, vec):
            score += w*v

          last = 1. / (1. + math.pow(math.e, -1*score)) 
          sampling.append( (last, res) )
      for last, res in sorted(sampling, key=lambda x:x[0]*-1):
        print(last, res)
        sample = m.parse(res.replace(" ", "")).strip()
        terms  = list( filter(lambda x:len(x) >= 4, map(lambda x:x.split("\t"), sample.split("\n") ) ) )
        break
