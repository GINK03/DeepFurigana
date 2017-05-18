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


""" ベースとなる文字列に対してどう変えたらもっともなろうチックなのか探索する """
def main():
    m = MeCab.Tagger("-Ochasen")
    term_freq = pickle.loads( open("vars/term_freq.pkl", "rb").read() )
     
    model = pickle.loads( open('vars/fasttext.gensim-model.pkl', 'rb').read() )
    
    ws = []
    with open("vars/linear.txt.model", "r") as f:
      next(f)
      next(f)
      next(f)
      next(f)
      next(f)
      next(f)
      for w in f:
        w = float(w.strip())
        ws.append(w)

    sample = m.parse("　１５年３月以降、日本や欧米、豪州で発火事故が計１６件あったが、けが人はいないという。製造工程で異物が混入したことなどが原因としている。").strip()

    terms  = list( filter(lambda x:len(x) >= 4, map(lambda x:x.split("\t"), sample.split("\n") ) ) )
    while True:
      sampling = []
      for e, term in enumerate(terms): 
        meta = term[3]
        if '名詞' not in meta:
          continue
        if '数' in meta:
          continue
        
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
        print(last, res, file=sys.stderr)
        sample = m.parse(res.replace(" ", "")).strip()
        terms  = list( filter(lambda x:len(x) >= 4, map(lambda x:x.split("\t"), sample.split("\n") ) ) )
        break

if __name__ == "__main__":
  main()
