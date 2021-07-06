import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, BertForTokenClassification, pipeline
from elasticsearch import Elasticsearch
import numpy as np


class Chatbot:
    mrc_model_path = './mrc'
    ner_model_path = './ner'

    def __init__(self):
        self.device = torch.device('cuda:0')
        self.es = Elasticsearch('localhost:9200')
        self.qa = self.get_qa_model()
        self.ner = self.get_ner_model()

    def get_qa_model(self):
        model = AutoModelForQuestionAnswering.from_pretrained(self.mrc_model_path)
        tokenizer = AutoTokenizer.from_pretrained(self.mrc_model_path)
        model.to(self.device)
        qa = pipeline("question-answering", model=model, tokenizer=tokenizer, device=0)
        return qa

    def get_ner_model(self):
        unique_tags = {'DT-B', 'DT-I', 'LC-B', 'LC-I', 'O', 'OG-B', 'OG-I', 'PS-B', 'PS-I', 'QT-B', 'QT-I', 'TI-B', 'TI-I'}

        self.tag2id = {tag: id for id, tag in enumerate(sorted(unique_tags))}
        self.id2tag = {id: tag for tag, id in self.tag2id.items()}

        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

        self.pad_token_id = self.tokenizer.pad_token_id
        self.cls_token_id = self.tokenizer.cls_token_id
        self.sep_token_id = self.tokenizer.sep_token_id
        self.pad_token_label_id = self.tag2id['O']
        self.cls_token_label_id = self.tag2id['O']
        self.sep_token_label_id = self.tag2id['O']

        ner = BertForTokenClassification.from_pretrained(self.ner_model_path, num_labels=len(unique_tags))
        ner.to(self.device)
        return ner

    def ner_tokenize(self, sent, max_seq_length):
        pre_syllable = "_"
        input_ids = [self.pad_token_id] * (max_seq_length - 1)
        attention_mask = [0] * (max_seq_length - 1)
        token_type_ids = [0] * max_seq_length
        sent = sent[:max_seq_length - 2]

        for i, syllable in enumerate(sent):
            if syllable == '_':
                pre_syllable = syllable
            if pre_syllable != "_":
                syllable = '##' + syllable
            pre_syllable = syllable

            input_ids[i] = (self.tokenizer.convert_tokens_to_ids(syllable))
            attention_mask[i] = 1

        input_ids = [self.cls_token_id] + input_ids
        input_ids[len(sent) + 1] = self.sep_token_id
        attention_mask = [1] + attention_mask
        attention_mask[len(sent) + 1] = 1
        return {"input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids}

    def ner_postprocess(self, ner_tagged_list):
        tmp = ''
        for txt, tag in ner_tagged_list:
            if tag != 'O':
                tmp += txt
            else:
                tmp += ' '

        result = tmp.replace('#', '').split()
        return ' '.join(result)

    def ner_infer(self, text):
        self.ner.eval()
        text = text.replace(' ', '_')

        predictions, true_labels = [], []

        tokenized_sent = self.ner_tokenize(text, len(text) + 2)
        input_ids = torch.tensor(tokenized_sent['input_ids']).unsqueeze(0).to(self.device)
        attention_mask = torch.tensor(tokenized_sent['attention_mask']).unsqueeze(0).to(self.device)
        token_type_ids = torch.tensor(tokenized_sent['token_type_ids']).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.ner(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids)

        logits = outputs[0]
        logits = logits.detach().cpu().numpy()
        label_ids = token_type_ids.cpu().numpy()

        predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
        true_labels.append(label_ids)

        pred_tags = [list(self.tag2id.keys())[p_i] for p in predictions for p_i in p]

        result = [(self.tokenizer.convert_ids_to_tokens(x), y) for x, y in zip(tokenized_sent['input_ids'], pred_tags)]
        result = self.ner_postprocess(result)
        return result

    def answer(self, question):
        query = {
            'query': {
                'bool': {
                    'must': [
                        {'match': {'text': question}}
                    ],
                    #  'should': [
                    #      {'match': {'text': self.ner_infer(question))}}
                    # ]
                }
            }
        }
        doc = self.es.search(index='document', body=query, size=2)['hits']['hits']
        
        ans_lst = []
        max_scr = doc[0]['_score']
        for i in range(len(doc)):
            ans = self.qa(question=question, context=doc[i]['_source']['text'], topk=1)
            if ans['answer'] == '' or 'unk' in ans['answer'] or len(ans['answer']) >= 30:
                pass
            else:
                ans_lst.append((ans['answer'], ans['score'] * doc[i]['_score'] / max_scr))
        
        res, res_score = sorted(ans_lst, key=lambda x: x[1], reverse=True)[0] # [0] # 1 : score -->
        return res, res_score
