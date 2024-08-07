# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
from query_pipeline import create_pipeline, query_llm

app = Flask(__name__)
CORS(app)

# Assuming the persist directory is specified
persist_dir = "/home/jason/coursistant/DR/llama-index/index/pdf"
pipeline = create_pipeline(persist_dir)

@app.route('/pdf_query', methods=['POST'])
def query():
    data = request.json
    user_input = data.get('query_str')
    if not user_input:
        return jsonify({"error": "query is required"}), 400
    response = query_llm(user_input, pipeline)
    return jsonify({
        "answer": response.answer,
        "file_name": response.file_name,
        "page_number": response.page_number,
        "image": response.image
    }), 200

if __name__ == '__main__':
    app.run(port=7000, debug=True)
