# flask API, Swagger UI

import text_preprocessing
from nltk.corpus import stopwords
import nltk
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import pickle
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import text_preprocessing as tp
from flask import request, Flask, jsonify

from flasgger import Swagger, LazyString, LazyJSONEncoder
from flasgger import swag_from

import pandas as pd

import sqlite3

############################################# START LSTM PRED ###################################################
import re
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# masukan model
loaded_model = load_model(r'model_of_lstm/lstm.h5')


# buat fungsi pred_sentiment
def pred_sentiment(text):
    # lakukan subtitusi jika karakter tidak termasuk di a-zA-Z0-9. maka dilakukan perubahan menjadi string kosong
    clean_text = re.sub(r'[^a-zA-Z0-9. ]', '', text)
    # ubah menjadi lowercase
    clean_text = clean_text.lower()
    # masukan ke variable text_new
    text_new = [clean_text]

    # Memecah kalimat menjadi kata.
    # saring text yang mengandung !"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n
    # ubah text menjadi lowercase
    tokenizer = Tokenizer(
        filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True)
    # setiap kalimat pada df['text'] diubah ke dalam bentuk tokenizer
    tokenizer.fit_on_texts(text_new)
    # ubah kata pada data df['text'] menjadi urutan integer
    sekuens_x = tokenizer.texts_to_sequences(text_new)
    # ubah urutan integer pada sequences_x ke dalam array 2D.
    # maxlen=None. maksimum panjang array akan disesuaikan dengan kalimat yang memiliki kumpulan kata terpanjang pada valiable sekuens_x
    # kalimat yang tidak memiliki kumpulan kata yang panjang akan di berikan pading
    padded_x = pad_sequences(sekuens_x, maxlen=None)

    # melakukan prediksi dengan menggunakan model yang telah dimuat. padded_x merupakan data yang akan di lakukan prediksi, batch_size adalah ukuran batch saat melakukan prediksi
    classes = loaded_model.predict(padded_x, batch_size=10)

    return classes[0]


# buat fungsi pred(classes)
def pred(classes):
    # lakukan pencetakan berdasarkan nilai maksimum dari array classes
    if classes[0] == classes.max():
        return 'negative'
    if classes[1] == classes.max():
        return 'neutral'
    if classes[2] == classes.max():
        return 'positive'


def text_to_pred(string):
    # panggil fungsi pred_sentiment dan masukan variabel string ke dalam fungsi
    classes = pred_sentiment(string)
    # panggil fungsi pred dan amsukan variable string
    result = pred(classes)
    # kembalikan hasil dari pred(classes)
    return result
################################################ END LSTM PRED ######################################################


################################################ START NN PRED ######################################################
# ori_text = "warung ini dimiliki oleh pengusaha pabrik tahu yang sudah puluhan tahun terkenal membuat tahu putih di bandung . tahu berkualitas , dipadu keahlian memasak , dipadu kretivitas , jadilah warung yang menyajikan menu utama berbahan tahu , ditambah menu umum lain seperti ayam . semuanya selera indonesia . harga cukup terjangkau . jangan lewatkan tahu bletoka nya , tidak kalah dengan yang asli dari tegal !"

# factory = StemmerFactory()
# stemmer = factory.create_stemmer()

# # load train dataset
# df_train = pd.read_csv(
#     'model_of_nn/train_preprocess.tsv', sep='\t', header=None)
# # df_train = pd.read_csv('/content/train_preprocess.tsv', sep='\t', header=None)
# df_data = df_train.rename(columns={0: 'Text', 1: 'Sentimen'})

# # load kamus alay dataset
# kamus_alay = pd.read_csv('model_of_nn/new_kamusalay.csv',
#                          encoding='latin-1', header=None)
# # kamus_alay = pd.read_csv('/content/new_kamusalay.csv', encoding='latin-1', header=None)
# kamus_alay = kamus_alay.rename(columns={0: 'original', 1: 'replacement'})

# id_stopword_dict = pd.read_csv('model_of_nn/stopwordbahasa.csv', header=None)
# # id_stopword_dict = pd.read_csv('/content/stopwordbahasa.csv', header=None)
# id_stopword_dict = id_stopword_dict.rename(columns={0: 'stopword'})


# Define the preprocessing function
# text cleansing with regex
def lowercase(text):
    return text.lower()


def hapus_karakter_ga_penting(text):
    text = re.sub('\n', ' ', text)  # Hapus enter
    text = re.sub('nya|deh|sih', ' ', text)  # Hapus stopwords tambahan
    text = re.sub('RT', ' ', text)  # Hapus RT
    text = re.sub('USER', ' ', text)  # Hapus USER
    # Hapus URL
    text = re.sub(
        '((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))', ' ', text)
    text = re.sub('  +', ' ', text)  # Hapus extra spaces
    text = re.sub('[^a-zA-Z0-9]', ' ', text)  # Hapus non huruf dan angka
    # Hapus non huruf atau apostrophe
    text = re.sub('\@[a-zA-Z0-9]*', ' ', text)
    # Hapus huruf tunggal
    text = ' '.join([w for w in text.split() if len(w) > 1])
    return text


def hapus_nonhurufangka(text):
    text = re.sub('[^0-9a-zA-Z]+', ' ', text)
    return text


def normalisasi_alay(text):

    # load train dataset
    df_train = pd.read_csv(
        'model_of_nn/train_preprocess.tsv', sep='\t', header=None)
    # df_train = pd.read_csv('/content/train_preprocess.tsv', sep='\t', header=None)
    df_data = df_train.rename(columns={0: 'Text', 1: 'Sentimen'})

    df_data_map = dict(zip(df_data['Text'], df_data['Sentimen']))

    return ' '.join([df_data_map[word] if word in df_data_map else word for word in text.split(' ')])


def hapus_stopword(text):
    id_stopword_dict = pd.read_csv(
        'model_of_nn/stopwordbahasa.csv', header=None)
    # id_stopword_dict = pd.read_csv('/content/stopwordbahasa.csv', header=None)
    id_stopword_dict = id_stopword_dict.rename(columns={0: 'stopword'})

    text = ' '.join(
        ['' if word in id_stopword_dict.stopword.values else word for word in text.split(' ')])
    text = re.sub('  +', ' ', text)  # Hapus extra spaces
    text = text.strip()
    return text


def stemming(text):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

    return stemmer.stem(text)


def cleansing(text):
    text = lowercase(text)
    text = hapus_nonhurufangka(text)
    text = hapus_karakter_ga_penting(text)
    text = normalisasi_alay(text)
    text = stemming(text)
    text = hapus_stopword(text)
    return text


count_vect = pickle.load(open("model_of_nn/feature_New3.sav", "rb"))
model_NN = pickle.load(open("model_of_nn/model_NN3.sav", "rb"))


# Download the Indonesian stopwords if not already downloaded

def text_to_pred_nn(text2):

    # contoh = "warung ini dimiliki oleh pengusaha pabrik tahu yang sudah puluhan tahun terkenal membuat tahu putih di bandung . tahu berkualitas , dipadu keahlian memasak , dipadu kretivitas , jadilah warung yang menyajikan menu utama berbahan tahu , ditambah menu umum lain seperti ayam . semuanya selera indonesia . harga cukup terjangkau . jangan lewatkan tahu bletoka nya , tidak kalah dengan yang asli dari tegal !"
    preprocessed_text = cleansing(text2)
    text_vector = count_vect.transform([preprocessed_text])

    result = model_NN.predict(text_vector)[0]
    return result
    # print("text : ", contoh)
    # print("\ntext_new : ", preprocessed_text)
    # print("\nSentiment : ", result)
################################################### END NN PRED #####################################################


app = Flask(__name__)

###############################################################################################################
app.json_encoder = LazyJSONEncoder

swagger_template = dict(
    info={
        'title': LazyString(lambda: 'API Documentation for Data Processing and Modeling'),
        'version': LazyString(lambda: '1.0.0'),
        'description': LazyString(lambda: 'Dokumentasi API untuk Data Processing dan Modeling')
    }, host=LazyString(lambda: request.host)
)

swagger_config = {
    "headers": [],
    "specs": [
        {
            "endpoint": 'docs',
            "route": '/docs.json'
        }
    ],
    "static_url_path": "/flasgger_static",
    "swagger_ui": True,
    "specs_route": "/docs/"
}

swagger = Swagger(app, template=swagger_template, config=swagger_config)
###############################################################################################################

# POST LSTM


@swag_from("docs/lstm_input.yml", methods=['POST'])
@app.route('/input_lstm', methods=['POST'])
def input_text_lstm():

    # masukan kalimat dengan key string
    string = request.get_json(force=True)

    # masukan data text ke variable string
    string = string['text']

    # masukan inputan variable string ke dalam function text_to_pred dan masukan hasil return ke variabel value_pred
    value_pred = text_to_pred(string)

    # buat dict untuk list text
    list_text = {'text': [], 'label': []}

    # masukan inputan variable string ke list_text['text']
    list_text['text'].append(string.lower())
    # masukan variable value_pred ke list_text['label']
    list_text['label'].append(value_pred)

    # koneksi ke database dengan nama sql
    conn = sqlite3.connect('sql1.db')

    # query untuk memasukan data ke table_pred
    query = '''insert into pred_table
                (text, label)
                values ('{}', '{}')'''.format(list_text['text'][0], list_text['label'][0])

    # eksekusi syntax query
    conn.execute(query)

    conn.commit()

    # ambil data dari pred_table
    df = pd.read_sql_query('SELECT * FROM pred_table', conn)

    # tutup koneksi
    conn.close()

    # masukan variable df ke fungsi to_dict
    dict_text = df.to_dict()

    # ubah variable dict_text ke json
    response_data = jsonify(dict_text)
    # kembalikan variable response_data
    return response_data
###############################################################################################################

# UPLOAD LSTM


@swag_from("docs/lstm_upload.yml", methods=['POST'])
@app.route('/upload_lstm', methods=['POST'])
def upload_text_lstm():

    # masukan file csv
    file = request.files['file']

    try:
        df1 = pd.read_csv(file, encoding='iso-8859-1', on_bad_lines='skip')
    except:
        df1 = pd.read_csv(file, encoding='utf-8', on_bad_lines='skip')

    # masukan data kolom ['Tweet'] ke dataframe
    df = pd.DataFrame(df1, columns=['Tweet'])

    # Untuk setiap data di Tweet, lakukan pengolahan dan masukan hasilnya ke df['label']
    df['label'] = df['Tweet'].apply(
        lambda x: text_to_pred(x))

    # ubah variable df ke dictionary
    json_data = df.to_dict()

    # koneksi ke database dengan nama sql1
    conn = sqlite3.connect('sql1.db')

    # inisialisasi varibale num
    num = 0

    # lakukan perulangan sebanyak jumlah data di json_data['Tweet']
    while num < len(json_data['Tweet']):

        # eksekusi syntax query
        conn.execute("insert into pred_table (text, label) values (?, ?)",
                     (json_data['Tweet'][num], json_data['label'][num]))

        # tambahkan 1 ke dalam variable num
        num = num + 1

    conn.commit()

    # ambil data dari pred_table
    df = pd.read_sql_query('SELECT * FROM pred_table', conn)

    # tutup koneksi
    conn.close()

    # masukan variable df ke fungsi to_dict
    dict_text = df.to_dict()

    # ubah variable dict_text ke json
    response_data = jsonify(dict_text)
    # kembalikan variable response_data
    return response_data

###############################################################################################################

# POST NN


@swag_from("docs/nn_input.yml", methods=['POST'])
@app.route('/input_nn', methods=['POST'])
def input_text_nn():

    # masukan kalimat dengan key string
    string = request.get_json(force=True)

    # masukan data text ke variable string
    string = string['text']

    # masukan inputan variable string ke dalam function text_to_pred_nn dan masukan hasil return ke variabel value_pred
    value_pred_nn = text_to_pred_nn(string)

    # buat dict untuk list text
    list_text = {'text': [], 'label': []}

    # masukan inputan variable string ke list_text['text']
    list_text['text'].append(string.lower())
    # masukan variable value_pred ke list_text['label']
    list_text['label'].append(value_pred_nn)

    # koneksi ke database dengan nama sql
    conn = sqlite3.connect('sql2.db')

    # query untuk memasukan data ke pred_NN
    query = '''insert into pred_NN
                (text, nSentiment)
                values ('{}', '{}')'''.format(list_text['text'][0], list_text['label'][0])

    # eksekusi syntax query
    conn.execute(query)

    conn.commit()

    # ambil data dari pred_table
    df = pd.read_sql_query('SELECT * FROM pred_NN', conn)

    # tutup koneksi
    conn.close()

    # masukan variable df ke fungsi to_dict
    dict_text = df.to_dict()

    # ubah variable dict_text ke json
    response_data = jsonify(dict_text)
    # kembalikan variable response_data
    return response_data
###############################################################################################################

# UPLOAD NN


@swag_from("docs/nn_upload.yml", methods=['POST'])
@app.route('/upload_nn', methods=['POST'])
def upload_text_nn():

    # masukan file csv
    file = request.files['file']

    try:
        df1 = pd.read_csv(file, encoding='iso-8859-1', on_bad_lines='skip')
    except:
        df1 = pd.read_csv(file, encoding='utf-8', on_bad_lines='skip')

    # masukan data kolom ['Tweet'] ke dataframe
    df = pd.DataFrame(df1, columns=['Tweet'])

    # Untuk setiap data di Tweet, lakukan pengolahan dan masukan hasilnya ke df['label']
    df['label'] = df['Tweet'].apply(
        lambda x: text_to_pred_nn(x))

    # ubah variable df ke dictionary
    json_data = df.to_dict()

    return json_data

    # koneksi ke database dengan nama sql1
    conn = sqlite3.connect('sql2.db')

    # inisialisasi varibale num
    num = 0

    # lakukan perulangan sebanyak jumlah data di json_data['Tweet']
    while num < len(json_data['Tweet']):

        # eksekusi syntax query
        conn.execute("insert into pred_NN (text, nSentiment) values (?, ?)",
                     (json_data['Tweet'][num], json_data['label'][num]))

        # tambahkan 1 ke dalam variable num
        num = num + 1

    conn.commit()

    # ambil data dari pred_NN
    df = pd.read_sql_query('SELECT * FROM pred_NN', conn)

    # tutup koneksi
    conn.close()

    # masukan variable df ke fungsi to_dict
    dict_text = df.to_dict()

    # ubah variable dict_text ke json
    response_data = jsonify(dict_text)
    # kembalikan variable response_data
    return response_data

###############################################################################################################


if __name__ == '__main__':
    app.run()
