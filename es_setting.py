# %%
# !pip install elasticsearch # 엘라스틱 서치를 파이썬에서 사용하기 위한 라이브러리 설치
# elasticsearch-7.0.0/bin/elasticsearch-plugin install analysis-nori # 한국어 토크나이저 설치
# elasticsearch-7.0.0/bin/elasticsearch-plugin install https://github.com/javacafe-project/elasticsearch-plugin/releases/download/v7.0.0/javacafe-analyzer-7.0.0.zip
# elasticsearch-7.0.0/bin/elasticsearch 실행

# %%
# 파이썬에 엘라스틱 서치 연결

from elasticsearch import Elasticsearch, helpers

es = Elasticsearch('localhost:9200', timeout=30, max_retries=10, retry_on_timeout=True)

print(es.info()) # 정상적으로 출력이 되면, 엘라스틱 서치 연결 완료

# %%
# 엘라스틱 서치에 문서를 저장할 index를 생성.
es.indices.create(index = 'document',
                  body = {
                      'settings':{
                          'analysis':{
                              'analyzer':{
                                  'my_analyzer':{
                                      "type": "custom",
                                      'tokenizer':'nori_tokenizer',
                                      'decompound_mode':'mixed',
                                      'stopwords':'_korean_',
                                      'synonyms':'_korean_',
                                      "filter": ["lowercase",
                                                 "my_shingle_f",
                                                 "nori_readingform",
                                                 "cjk_bigram",
                                                 "decimal_digit",
                                                 "stemmer",
                                                 "trim"]
                                  },
                                  'kor2eng_analyzer':{
                                      'type':'custom',
                                      'tokenizer':'nori_tokenizer',
                                      'filter': [
                                          'trim',
                                          'lowercase',
                                          'javacafe_kor2eng'
                                      ]
                                  },
                                  'eng2kor_analyzer': {
                                      'type': 'custom',
                                      'tokenizer': 'nori_tokenizer',
                                      'filter': [
                                          'trim',
                                          'lowercase',
                                          'javacafe_eng2kor'
                                      ]
                                  },
                              },
                              'filter':{
                                  'my_shingle_f':{
                                      "type": "shingle"
                                  }
                              }
                          },
                          'similarity':{
                              'my_similarity':{
                                  'type':'BM25',
                              }
                          }
                      },
                      'mappings':{
                          'properties':{
                              'title':{
                                  'type':'keyword',
                                  'copy_to':['title_kor2eng','title_eng2kor']
                              },
                              'title_kor2eng': {
                                  'type': 'text',
                                  'analyzer':'my_analyzer',
                                  'search_analyzer': 'kor2eng_analyzer'
                              },
                              'title_eng2kor': {
                                  'type': 'text',
                                  'analyzer':'my_analyzer',
                                  'search_analyzer': 'eng2kor_analyzer'
                              },
                              'text':{
                                  'type':'text',
                                  'analyzer':'my_analyzer',
                                  'similarity':'my_similarity',
                              },
                              'text_origin': {
                                  'type': 'text',
                                  'analyzer': 'my_analyzer',
                                  'similarity': 'my_similarity'
                              }
                          }
                      }
                  }
                  )

# %%
# 정상적으로 출력이 되면, index 생성 완료
print(es.indices.get('document'))

# %%

import json

with open('data/namuwiki_baseball.json', 'r', encoding='utf-8') as f:
    wiki_data = json.load(f)

# %%

from tqdm import tqdm

wiki_data_title = list(wiki_data.keys())
wiki_data_text = list(wiki_data.values())

title = []
text = []

for num in tqdm(range(len(wiki_data_title))):
    cnt = 0
    while cnt < len(wiki_data_text[num]):
        title.append(wiki_data_title[num])
        text.append(wiki_data_text[num][cnt:cnt+1000])
        cnt+=1000

# %%

import pandas as pd

df = pd.DataFrame({'title' : title,'text' : text})

buffer = []
rows = 0

for num in tqdm(range(len(df))):
    article = {"_id": num,
               "_index": "document",
               "title" : df['title'][num],
               "text" : df['text'][num]}

    buffer.append(article)

    rows += 1

    if rows % 3000 == 0:
        helpers.bulk(es, buffer)
        buffer = []

        print("Inserted {} articles".format(rows), end="\r")

if buffer:
    helpers.bulk(es, buffer)

print("Total articles inserted: {}".format(rows))
