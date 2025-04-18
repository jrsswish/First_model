from flask import Flask, request, render_template
import pickle
from model import predict

app = Flask(__name__)

# load our pickle model
model = pickle.load(open("model.pkl", "rb"))

# get reuqest which just returns the html state
@app.route("/")
def home():
    return render_template("index.html")

# post request which uses the predictor function from model.py
@app.route("/predict", methods=["POST"])
def predict_model():
    # requests for degree and age
    degree = request.form.get("degree")
    age = request.form.get("age")

    # uses those values to use my function
    result = predict(degree, age)

    # return prediction_text as result then output the prediction_text
    return render_template("index.html", prediction_text=result)

if __name__ == "__main__":
    app.run(port=3000, debug=True)