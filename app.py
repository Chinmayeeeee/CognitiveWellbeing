from flask import Flask, request, jsonify
import joblib
import numpy as np
import sentimentForAudiorec

model = joblib.load('svm.pkl')

# Flask Constructor
app = Flask(__name__)


# decorator to associate
# a function with the url
@app.route("/")
def showHomePage():
    # response from the server
    return "This is home page"

@app.route('/predict', methods=['POST'])
def predict():
    test_data = request.form["test_data"]
    input= [test_data][0]
    output = sentimentForAudiorec.test(input)

    # output = model.predict(input)[0]
    # return jsonify({'Prediction': str(output)})
    return str(output)


if __name__ == "__main__":
    app.run(debug=True)
    # app.run(host="0.0.0.0")