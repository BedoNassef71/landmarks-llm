from flask import Flask
from routes import get_class_info, get_question

app = Flask(__name__)

@app.route('/info', methods=['POST'])
def info():
    return get_class_info()

@app.route('/chat', methods=['POST'])
def chat():
    return get_question()

if __name__ == '__main__':
    app.run(debug=True)
