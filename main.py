import pandas as pd

from flask import Flask

from textblob import TextBlob
from deep_translator import GoogleTranslator

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv('data/casas.csv')

colunas = ['tamanho', 'preco']
df = df.loc[:, colunas]

X = df.drop('preco', axis=1)
y = df.loc[:, ['preco']]

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3, random_state=42)

modelo = LinearRegression()
modelo.fit(X_train, Y_train['preco'])

app = Flask(__name__)

@app.route('/')
def home():

    return "Minha primeira API."

@app.route('/sentimento/<frase>')
def sentimento(frase):

    gt = GoogleTranslator(source='pt', target='en')

    frase_en = gt.translate(frase)

    tb = TextBlob(frase_en)

    polaridade = tb.sentiment.polarity

    return f"polaridade: {polaridade}"

@app.route('/cotacao/<int:tamanho>')
def cotacao(tamanho):

    preco = modelo.predict([[tamanho]])

    return str(preco)

app.run(debug=True)