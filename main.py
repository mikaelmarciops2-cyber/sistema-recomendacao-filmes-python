import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Ler dados
df = pd.read_csv("filmes.csv")

# Converter texto em números
vectorizer = CountVectorizer()

matriz = vectorizer.fit_transform(df["genero"])

# Calcular similaridade
similaridade = cosine_similarity(matriz)

# Escolher filme
filme_usuario = input("Digite um filme: ")

# Verificar se existe
if filme_usuario in df["filme"].values:

    indice = df[df["filme"] == filme_usuario].index[0]

    scores = list(enumerate(similaridade[indice]))

    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    print("\nFilmes recomendados:\n")

    for i in scores[1:4]:

        print(df.iloc[i[0]]["filme"])

else:
    print("Filme não encontrado.")