from flask import request, jsonify
from utils import ans_question, user_input

def get_class_info():
    data = request.get_json()
    class_name = data['class_name']
    result = user_input(class_name)
    return jsonify(result)

def get_question():
    data = request.get_json()
    question = data['question']
    result = ans_question(question)
    return jsonify(result)
