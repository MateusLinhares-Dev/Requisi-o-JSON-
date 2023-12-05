from ml_request import treinamento_ml
from flask import Flask, request


app = Flask(__name__)


@app.route('/')
def hello():
    return "<h1>Index</h1>"

@app.route('/prever', methods=['POST'])
def get_and_http():
    data = request.json
    model, le= treinamento_ml()

    industry = data.get("industry", [])

    predicoes = []
    for industry_item in industry:
        predicao = model.predict([industry_item])
        predicao_origin = le.inverse_transform([predicao])[0]
        predicoes.append({'industry': industry_item, 'predicao': predicao_origin})

    return {'predicoes': predicoes}



if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8080)