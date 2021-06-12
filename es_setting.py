# %%
# !pip install elasticsearch # 엘라스틱 서치를 파이썬에서 사용하기 위한 라이브러리 설치
# elasticsearch-7.0.0/bin/elasticsearch-plugin install analysis-nori # 한국어 토크나이저 설치
# elasticsearch-7.0.0/bin/elasticsearch-plugin install https://github.com/javacafe-project/elasticsearch-plugin/releases/download/v7.0.0/javacafe-analyzer-7.0.0.zip
# elasticsearch-7.0.0/bin/elasticsearch 실행

# %%
# 파이썬에 엘라스틱 서치 연결

from elasticsearch import Elasticsearch

es = Elasticsearch('localhost:9200', timeout=30, max_retries=10, retry_on_timeout=True)

print(es.info()) # 정상적으로 출력이 되면, 엘라스틱 서치 연결 완료

# %%
# 엘라스틱 서치에 문서를 저장할 index를 생성.
# es.indices.create(index = 'document',
#                   body = {
#                       'settings':{
#                           'analysis':{
#                               'analyzer':{
#                                   'my_analyzer':{
#                                       "type": "custom",
#                                       'tokenizer':'nori_tokenizer',
#                                       'decompound_mode':'mixed',
#                                       'stopwords':'_korean_',
#                                       'synonyms':'_korean_',
#                                       "filter": ["lowercase",
#                                                  "my_shingle_f",
#                                                  "nori_readingform",
#                                                  "cjk_bigram",
#                                                  "decimal_digit",
#                                                  "stemmer",
#                                                  "trim"]
#                                   },
#                                   'kor2eng_analyzer':{
#                                       'type':'custom',
#                                       'tokenizer':'nori_tokenizer',
#                                       'filter': [
#                                           'trim',
#                                           'lowercase',
#                                           'javacafe_kor2eng'
#                                       ]
#                                   },
#                                   'eng2kor_analyzer': {
#                                       'type': 'custom',
#                                       'tokenizer': 'nori_tokenizer',
#                                       'filter': [
#                                           'trim',
#                                           'lowercase',
#                                           'javacafe_eng2kor'
#                                       ]
#                                   },
#                               },
#                               'filter':{
#                                   'my_shingle_f':{
#                                       "type": "shingle"
#                                   }
#                               }
#                           },
#                           'similarity':{
#                               'my_similarity':{
#                                   'type':'BM25',
#                               }
#                           }
#                       },
#                       'mappings':{
#                           'properties':{
#                               'title':{
#                                   'type':'keyword',
#                                   'copy_to':['title_kor2eng','title_eng2kor']
#                               },
#                               'title_kor2eng': {
#                                   'type': 'text',
#                                   'analyzer':'my_analyzer',
#                                   'search_analyzer': 'kor2eng_analyzer'
#                               },
#                               'title_eng2kor': {
#                                   'type': 'text',
#                                   'analyzer':'my_analyzer',
#                                   'search_analyzer': 'eng2kor_analyzer'
#                               },
#                               'text':{
#                                   'type':'text',
#                                   'analyzer':'my_analyzer',
#                                   'similarity':'my_similarity',
#                               },
#                               'text_origin': {
#                                   'type': 'text',
#                                   'analyzer': 'my_analyzer',
#                                   'similarity': 'my_similarity'
#                               }
#                           }
#                       }
#                   }
#                   )

# %%
# 정상적으로 출력이 되면, index 생성 완료
print(es.indices.get('document'))

# %%
# 엘라스틱 서치에 팀 문서를 추가
import os

for file_name in os.listdir('data/club'):
    if file_name != '.DS_Store':
        with open('data/club/'+file_name, 'r') as f:
            contents = f.read()
        es.index(index='document', body = {"title" : file_name.split('.')[0], "text" : contents})

# %%
# 엘라스틱 서치에 선수 문서를 추가
import os

for file_name in os.listdir('data/player'):
    if file_name != '.DS_Store':
        with open('data/player/'+file_name, 'r') as f:
            contents = f.read()
        es.index(index='document', body = {"title" : file_name.split('.')[0], "text" : contents})
