# app_piazza.py
from flask import Flask, request, jsonify
from flask_cors import CORS
from query_pipeline import create_pipeline, query_llm  # Ensure this matches the Piazza pipeline module

app = Flask(__name__)
CORS(app)

# Assuming the persist directory is specified for Piazza
persist_dir = "/home/jason/coursistant/DR/llama-index/index/piazza"
pipeline = create_pipeline(persist_dir)

@app.route('/piazza_query', methods=['POST'])
def query():
    data = request.json
    user_input = data.get('query_str')
    if not user_input:
        return jsonify({"error": "query is required"}), 400
    response = query_llm(user_input, pipeline)
    return jsonify({
        "answer": response.answer,
    }), 200

if __name__ == '__main__':
    app.run(port=7001, debug=True)
