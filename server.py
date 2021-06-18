from flask import Flask, request, jsonify
from chatbot import Chatbot
from time import time


app = Flask(__name__)


@app.route('/keyboard')
def Keyboard():
    dataSend = {}
    return jsonify(dataSend)


@app.route('/message', methods=['POST'])
def Message():
    global bot

    content = request.get_json()
    question = content['userRequest']['utterance'].strip()
    print('question:', question)

    prev = time()
    answer, score = bot.answer(question)
    now = time()
    print('inference time:', now - prev)

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
    bot = Chatbot()
    app.run(host='0.0.0.0', port=6006)
