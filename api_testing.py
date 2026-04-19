from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    \"\"\"
    A simple backend endpoint to fulfill the API testing (Postman) criteria.
    It expects a JSON payload containing numerical movie metrics and simulates a Random Forest regression response.
    \"\"\"
    try:
        # Get JSON data sent from Postman
        data = request.get_json()
        
        # Check if no payload was sent
        if not data:
            return jsonify({\"error\": \"Please provide a JSON payload. Example: {'budget': 1000000, 'popularity': 150, 'revenue': 20000000, 'runtime': 120, 'vote_count': 500}\"}), 400
            
        # Example validation for expected fields
        expected_fields = ['budget', 'popularity', 'revenue', 'runtime', 'vote_count']
        missing_fields = [f for f in expected_fields if f not in data]
        
        if missing_fields:
            return jsonify({\"error\": f\"Missing features in payload: {missing_fields}\"}), 400
            
        # Simulating a prediction using the payload parameters
        # In a real deployed environment, here we would call model.predict()
        # For academic demonstration, we return a mock generated rating.
        computed_rating = 6.4 + (int(data['popularity']) * 0.005) + (int(data['runtime']) * 0.001)
        predicted_vote_average = min(round(computed_rating, 1), 10.0) # Ensure it doesn't exceed 10.0
        
        response = {
            \"status\": \"success\",
            \"message\": \"Data processed successfully.\",
            \"input_received\": data,
            \"predicted_vote_average\": predicted_vote_average
        }
        
        return jsonify(response), 200

    except Exception as e:
        return jsonify({\"error\": str(e)}), 500

@app.route('/')
def home():
    return jsonify({
        \"message\": \"API is running! Use Postman to send a POST request to /predict\",
        \"status\": \"active\"
    })

if __name__ == '__main__':
    # Run the app to listen locally
    app.run(host='0.0.0.0', port=5000, debug=True)
