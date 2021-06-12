from flask import Flask, request, jsonify
from chatbot.inference import get_elastics, get_answer, init_inference
from time import time
app = Flask(__name__)

global es, qa, ner_model, sep_token_id

@app.route('/keyboard')
def Keyboard():
    dataSend = {
    }
    return jsonify(dataSend)


@app.route('/message', methods=['POST'])
def Message():
    content = request.get_json()
    text = content['userRequest']['utterance'].strip()
    print(text)
    global es, qa, ner_model, sep_token_id
    prev = time()
    ans, score = get_answer(text, es, qa, ner_model)
    now = time()
    print(now - prev)
    answer = ans + str(score)
    dataSend = {
            "version": "2.0",
            "template": {
                "outputs": [
                    {
                        "simpleText": {
                            "text": answer
                        }
                    }
                ]
            }
        }
    
    return jsonify(dataSend)

if __name__ == "__main__":
    global es, qa, ner_model, sep_token_id
    qa, ner_model = init_inference()
    es = get_elastics()
    app.run(host = '0.0.0.0', port = 6006)

