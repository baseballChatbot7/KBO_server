# %%
# 미리 load 되어야 하는 부분

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, BertForTokenClassification, pipeline


global pad_token_id, cls_token_id, sep_token_id, pad_token_label_id, cls_token_label_id, sep_token_label_id
global tokenizer, device
global tag2id, id2tag

# NER Setting
class TokenDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val) for key, val in self.encodings[idx].items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def ner_tokenizer(sent, max_seq_length):
    global pad_token_id, cls_token_id, sep_token_id
    global tokenizer

    pre_syllable = "_"
    input_ids = [pad_token_id] * (max_seq_length - 1)
    attention_mask = [0] * (max_seq_length - 1)
    token_type_ids = [0] * max_seq_length
    sent = sent[:max_seq_length - 2]

    for i, syllable in enumerate(sent):
        if syllable == '_':
            pre_syllable = syllable
        if pre_syllable != "_":
            syllable = '##' + syllable
        pre_syllable = syllable

        input_ids[i] = (tokenizer.convert_tokens_to_ids(syllable))
        attention_mask[i] = 1

    input_ids = [cls_token_id] + input_ids
    input_ids[len(sent) + 1] = sep_token_id
    attention_mask = [1] + attention_mask
    attention_mask[len(sent) + 1] = 1
    return {"input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids}


def train_ner_inference(text, model_ner):
    global device
    global tag2id, id2tag

    model_ner.eval()
    text = text.replace(' ', '_')

    predictions, true_labels = [], []

    tokenized_sent = ner_tokenizer(text, len(text) + 2)
    input_ids = torch.tensor(tokenized_sent['input_ids']).unsqueeze(0).to(device)
    attention_mask = torch.tensor(tokenized_sent['attention_mask']).unsqueeze(0).to(device)
    token_type_ids = torch.tensor(tokenized_sent['token_type_ids']).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model_ner(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids)

    logits = outputs[0]
    logits = logits.detach().cpu().numpy()
    label_ids = token_type_ids.cpu().numpy()

    predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
    true_labels.append(label_ids)

    pred_tags = [list(tag2id.keys())[p_i] for p in predictions for p_i in p]

    return [(tokenizer.convert_ids_to_tokens(x), y) for x, y in zip(tokenized_sent['input_ids'], pred_tags)]


def ner_return(ner_tagged_list):
    tmp = ''
    for txt, tag in ner_tagged_list:
        if tag != 'O':
            tmp += txt
        else:
            tmp += ' '

    result = tmp.replace('#', '').split()

    return ' '.join(result)

def init_inference():
    global pad_token_id, cls_token_id, sep_token_id, pad_token_label_id, cls_token_label_id, sep_token_label_id
    global tokenizer, device
    global tag2id, id2tag
    unique_tags = {'DT-B', 'DT-I', 'LC-B', 'LC-I', 'O', 'OG-B', 'OG-I', 'PS-B', 'PS-I', 'QT-B', 'QT-I', 'TI-B', 'TI-I'}

    tag2id = {tag: id for id, tag in enumerate(sorted(unique_tags))}
    id2tag = {id: tag for tag, id in tag2id.items()}

    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

    pad_token_id = tokenizer.pad_token_id
    cls_token_id = tokenizer.cls_token_id
    sep_token_id = tokenizer.sep_token_id
    pad_token_label_id = tag2id['O']
    cls_token_label_id = tag2id['O']
    sep_token_label_id = tag2id['O']

    #model_ner = BertForTokenClassification.from_pretrained(r'./chatbot/ner', num_labels=len(unique_tags))

    device = torch.device('cuda:0')

    #model_ner.to(device)
    #  /NER

    # MRC Setting

    model = AutoModelForQuestionAnswering.from_pretrained("./chatbot/mrc")
    tokenizer = AutoTokenizer.from_pretrained("./chatbot/mrc")

    model.to(device)

    qa = pipeline("question-answering", model=model, tokenizer=tokenizer, device=0)

    return qa, None#model_ner

def get_elastics():

    # /MRC

    # %%
    # Inference하는 부분

    from elasticsearch import Elasticsearch

    es = Elasticsearch('localhost:9200')

    return es

def get_answer(question, es, qa, ner_model):
    query = {
        'query': {
            'bool': {
                'must': [
                    {'match': {'text': question}}
                ],
                #  'should': [
                #      {'match': {'text': ner_return(train_ner_inference(question, ner_model))}}
                # ]
            }
        }
    }
    doc = es.search(index='document', body=query, size=2)['hits']['hits']
    
    ans_lst = []
    max_scr = doc[0]['_score']
    for i in range(len(doc)):
        ans = qa(question=question, context=doc[i]['_source']['text'], topk=1)
        if ans['answer'] == '' or 'unk' in ans['answer'] or len(ans['answer']) >= 30:
            pass
        else:
            ans_lst.append((ans['answer'], ans['score'] * doc[i]['_score'] / max_scr))
    
    res , res_score = sorted(ans_lst, key=lambda x: x[1], reverse=True)[0] # [0] # 1 : score -->
    return res, res_score