import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline



#tratamento de dados da aprendizagem e treinamento do modelo
def treinamento_ml():
    df_treinamento = pd.read_csv('treinamento/treinarIA.csv', encoding='utf-8')
    df_treinamento['Industry'] = df_treinamento['Industry'].fillna('Services and Consulting').astype('str')

    x_train, x_test, y_train, y_test = train_test_split(
        df_treinamento['Industry'],
        df_treinamento['Industria-correto'],
        test_size=0.2,
        random_state=42
    )

   # Treinamento do modelo
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)

    # Use CountVectorizer apenas nos dados de texto
    text_model = make_pipeline(CountVectorizer(), MultinomialNB())
    text_model.fit(x_train.astype(str), y_train_encoded)

    return text_model, le

if __name__ == "__main__":
    treinamento_ml()

