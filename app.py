from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)

# Load data from CSV
data = pd.read_csv("cmrit_data.csv")

# Prepare TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['question'])

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message")
    if not user_input:
        return jsonify({"response": "Please enter a valid query."})

    # Vectorize user input
    user_vec = vectorizer.transform([user_input])

    # Compute similarity with dataset
    similarities = cosine_similarity(user_vec, X)
    best_match_index = similarities.argmax()
    best_score = similarities[0, best_match_index]

    # Threshold to avoid irrelevant answers
    if best_score < 0.3:
        return jsonify({"response": "Sorry, I couldn't find an answer. Please try rephrasing."})
    
    response = data.iloc[best_match_index]['answer']
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)





































# from flask import Flask, request, jsonify
# from flask_cors import CORS

# app = Flask(__name__)
# CORS(app)

# # Sample response logic

# @app.route("/chat", methods=["POST"])
# def chat():
#     data = request.get_json()
#     print("Received data:", data)  # 👈 Add this line
#     user_message = data.get("message", "").lower()

#     if "admission" in user_message:
#         return jsonify({"response": "CMRIT admissions are open! Visit https://www.cmrit.ac.in/admissions for more info."})
#     elif "courses" in user_message:
#         return jsonify({"response": "CMRIT offers B.Tech, M.Tech, MBA, and Ph.D programs."})
#     elif "hostel" in user_message:
#         return jsonify({"response": "Yes, CMRIT has separate hostels for boys and girls with all basic amenities."})
#     else:
#         return jsonify({"response": "Sorry, I don't have information on that yet. Please contact the college office."})


# if __name__ == "__main__":
#     app.run(debug=True)

