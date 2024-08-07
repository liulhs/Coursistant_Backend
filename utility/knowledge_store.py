from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import os
import json

app = Flask(__name__)
CORS(app)  # This will allow CORS for all routes

@app.route('/store-piazza', methods=['POST'])
def store_piazza():
    data = request.get_json()
    Piazza_course_id = data.get('Piazza_course_id')
    course_id = data.get('course_id')
    email = data.get('email')

    if not Piazza_course_id or not course_id or not email:
        return jsonify({"error": "Missing required parameters"}), 400

    try:
        # Make the GET request to fetch all posts for the course
        response = requests.get(f'https://piazza.e-ta.net/users/{email}/courses/{Piazza_course_id}/posts/all')
        response.raise_for_status()

        # Create the directory path
        dir_path = os.path.join('/home/jason/coursistant/Coursistant_Backend/database', str(course_id), 'Piazza')
        file_path = os.path.join(dir_path, 'all_posts.json')

        # Check if the directory exists, if not create it
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)

        # Write the response data to the file, replacing if it already exists
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(response.json(), f, ensure_ascii=False, indent=2)

        return jsonify({"message": "Piazza knowledge successfully stored."}), 200

    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Failed to fetch posts: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": f"Failed to store Piazza knowledge: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(port=5600)
