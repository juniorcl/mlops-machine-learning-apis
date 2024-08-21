import pickle

from flask import Flask, request, jsonify
from flask_basicauth import BasicAuth

from textblob import TextBlob
from deep_translator import GoogleTranslator

from sklearn.linear_model import LinearRegression


with open('./models/modelo.sav', 'rb') as f:
    modelo = pickle.load(f)

colunas = ['tamanho', 'ano', 'garagem']

app = Flask(__name__)

app.config['BASIC_AUTH_USERNAME'] = 'julio'
app.config['BASIC_AUTH_PASSWORD'] = 'alura'

basic_auth = BasicAuth(app)

@app.route('/')
def home():

    return "Minha primeira API."

@app.route('/sentimento/<frase>')
@basic_auth.required
def sentimento(frase):

    gt = GoogleTranslator(source='pt', target='en')

    frase_en = gt.translate(frase)

    tb = TextBlob(frase_en)

    polaridade = tb.sentiment.polarity

    return f"polaridade: {polaridade}"

@app.route('/cotacao/', methods=['POST'])
def cotacao():

    dados = request.get_json()

    dados_input = [dados[col] for col in colunas]

    preco = modelo.predict([dados_input])

    return jsonify(preco=preco[0])

app.run(debug=True)