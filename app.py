from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

model = pickle.load(open('model.pkl','rb'))

@app.route("/")
def home():
    return render_template('index.html')


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method=="POST":
        a = float(request.form["Sepal_Length"])
        b = float(request.form["Sepal_Width"])
        c = float(request.form["Petal_Length"])
        d = float(request.form["Petal_Width"])
        int_features = [a, b, c, d]
        final_features = np.array(int_features).reshape(1, -1)
        prediction = model.predict(final_features)
        output = prediction[0]
        if output==0:
            results="The Expected Leaf Name Will Be : Iris-Setosa"
        elif output==1:
            results="The Expected Leaf Name Will Be : Iris-Versicolor"
        else:
            results="The Expected Leaf Name Will Be : Iris-Virginica"
        return render_template("index.html", prediction_text=results)
    else:
        return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True)
