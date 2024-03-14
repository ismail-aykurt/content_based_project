# Content Based Recommendation (İçerik Temelli Sınıflandırma)

# Film Overviewlerine göre tavsiye sistemi oluşturma
# 1: TF-IDF matrislerini oluştur
# 2: Cosine Similarity(Kosinüs benzerliği) matrisinin oluşturulması
# 3: Benzerliklerine göre önerilerin yapılması
# 4: Scriptin hazırlanması

import pandas as pd
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
pd.set_option("display.expand_frame_repr", False)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("the_movies_dataset/movies_metadata.csv", low_memory=False)
df.head()
df.shape
df.isnull().values.any()
df.isnull().sum()
df.describe().T

tfidf = TfidfVectorizer(stop_words="english")  # Burda ingilizcede sıklıkla geçen kelimeleri sildik mesela and in on vs
# Bunları neden sildik dersek de bunlar sık kullanılan kelimeler oldugu için bize yanlış sonuçlar döndğrecektir

df["overview"].isnull().sum()  # Evet burda da boş değerler oldugunu gördük bunalrı  ayarlayalım

df["overview"] = df["overview"].fillna("")  # Boş değerleri boşluk tamamem boşalltık orda veri yok
tfidf_matrix = tfidf.fit_transform(df["overview"])
tfidf_matrix.shape

tfidf.get_feature_names_out()

tfidf_matrix.toarray()

cos_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

indices = pd.Series(df.index, index=df["title"])
indices.head()
indices.index.value_counts()
# şimdi de burda bir sıkıntı var bir film ismi birden fazla çoklanmış bunları tek haneye indirmek gerekiyor
# Bu problemi ortadan kaldırmak için duplicates() metodunu kullancaz ve en son film yani en güncel olanı alcaz

indices = indices[~indices.index.duplicated(keep="last")]
# Bu kod satırı gider bakar duplice eden isimler var mı eğer varsa True döner ancak biz olmayanları alalım dedik ~ ile

indices["Cinderella"]
indices["Hamlet"]  # Evet önceden çoklamalı olan film isimleri şimdi tek oldu ve en güncel olanı aldık

movie_index = indices["Sherlock Holmes"]
cos_sim[movie_index]

similarity_scores = pd.DataFrame(cos_sim[movie_index],
                                 columns=["score"])

movie_indices = similarity_scores.sort_values("score", ascending=False)[1:11].index

df["title"].iloc[movie_indices]


def content_based_recommender(title, cosine_sim, dataframe):
    # index'leri olusturma
    indices = pd.Series(dataframe.index, index=dataframe['title'])
    indices = indices[~indices.index.duplicated(keep='last')]
    # title'ın index'ini yakalama
    movie_index = indices[title]
    # title'a gore benzerlik skorlarını hesapalama
    similarity_scores = pd.DataFrame(cosine_sim[movie_index], columns=["score"])
    # kendisi haric ilk 10 filmi getirme
    movie_indices = similarity_scores.sort_values("score", ascending=False)[1:11].index
    return dataframe['title'].iloc[movie_indices]

def calculate_cosine_sim(dataframe):
    tfidf = TfidfVectorizer(stop_words='english')
    dataframe['overview'] = dataframe['overview'].fillna('')
    tfidf_matrix = tfidf.fit_transform(dataframe['overview'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim

cosine_sim = calculate_cosine_sim(df)
content_based_recommender('The Dark Knight Rises', cosine_sim, df)







