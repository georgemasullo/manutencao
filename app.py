from flask import Flask, request ,jsonify
from flask import make_response
from flask import abort
from flask_httpauth import HTTPBasicAuth
import nltk

app = Flask(__name__)
auth = HTTPBasicAuth()

arq = open('base.txt')
linha = arq.readlines()
arq.close()
base=[]
#print(linha)
for i in range(0,len(linha)):
    l=linha[i].split('|')
    base.append((l[0],l[1]))

stopwordsnltk = nltk.corpus.stopwords.words('portuguese')
stopwordsnltk.append('vou')
stopwordsnltk.append('tao')
#print(stopwordsnltk)

def removestopwords(texto):
    frases = []
    for (palavras, emocao) in texto:
        semstop = [p for p in palavras.split() if p not in stopwordsnltk]
        frases.append((semstop, emocao))
    return frases

#print(removestopwords(base))

def aplicastemmer(texto):
    stemmer = nltk.stem.RSLPStemmer()
    frasessstemming = []
    for (palavras, emocao) in texto:
        comstemming = [str(stemmer.stem(p)) for p in palavras.split() if p not in stopwordsnltk]
        frasessstemming.append((comstemming, emocao))
    return frasessstemming

frasescomstemmingtreinamento = aplicastemmer(base)
#print(frasescomstemming)

def buscapalavras(frases):
    todaspalavras = []
    for (palavras, emocao) in frases:
        todaspalavras.extend(palavras)
    return todaspalavras

palavrastreinamento = buscapalavras(frasescomstemmingtreinamento)

def buscafrequencia(palavras):
    palavras = nltk.FreqDist(palavras)
    return palavras

frequenciatreinamento = buscafrequencia(palavrastreinamento)
#print(frequencia.most_common(50))

def buscapalavrasunicas(frequencia):
    freq = frequencia.keys()
    return freq

palavrasunicastreinamento = buscapalavrasunicas(frequenciatreinamento)
#print(palavrasunicastreinamento)

#print(palavrasunicas)

def extratorpalavras(documento):
    doc = set(documento)
    caracteristicas = {}
    for palavras in palavrasunicastreinamento:
        caracteristicas['%s' % palavras] = (palavras in doc)
    return caracteristicas

basecompletatreinamento = nltk.classify.apply_features(extratorpalavras, frasescomstemmingtreinamento)

classificador = nltk.NaiveBayesClassifier.train(basecompletatreinamento)
#print(classificador.labels())
#print(classificador.show_most_informative_features(20))

#@app.route('/')
#def hello():
#    return 'Hello World!

@auth.verify_password
def verify_password(username, password):
    if username == 'george' and password == '123' :
        return True
    return False
@auth.error_handler
def unauthorized():
    return make_response(jsonify({'error': 'Unauthorized access'}), 401)
@app.route('/classificador/api/v1.0/frases', methods=['GET'])
@auth.login_required
def classificar_frase():
    if not request.json or not 'frase' in request.json:
        abort(404)
    frase=request.json['frase']
    #teste = 'eu sinto amor por voce'
    testestemming = []
    stemmer = nltk.stem.RSLPStemmer()
    for (palavrastreinamento) in frase.split():
        comstem = [p for p in palavrastreinamento.split()]
        testestemming.append(str(stemmer.stem(comstem[0])))
    #print(testestemming)

    novo = extratorpalavras(testestemming)
    #print(novo)

    cl=classificador.classify(novo)
    distribuicao = classificador.prob_classify(novo)
    return jsonify({'Classificacao': cl  , 'dados' : [(classe,distribuicao.prob(classe))for classe in distribuicao.samples()]})
@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)
if __name__ == '__main__':
     app.run(host="0.0.0.0",port=80)
