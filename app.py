from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np 

app = Flask(__name__)

@app.route("/", methods=['GET', 'POST'])
def weight_prediction():
    if request.method == 'GET':
        return render_template('weight-prediction.html')
    elif request.method == 'POST':
        print(dict(request.form))
        weight_features = dict(request.form).values()
        weight_features = np.array([int(x) for x in weight_features])
        
        model = joblib.load('model-development/weight-predict.joblib')
        transform_array = np.reshape(weight_features, (1,-1))
        # get_gender = transform_array[[0]] 
        # get_height = transform_array[[1]]
        gender = {
            '0':'Female',
            '1':'Male'
        }
        
        prediction = model.predict(transform_array)
        
        result = int(prediction)
        return render_template('weight-prediction.html', result=result, get_gender=gender.get(str(weight_features[1])), get_height=str(weight_features[0]))

    else:
        return "Unsupported Request Method!"

if __name__ == "__main__":
    app.run(port=5000, debug=True)
