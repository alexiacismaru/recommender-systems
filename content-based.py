import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

metadata = pd.read_csv('top_brands_cosmetics_product_reviews.csv', low_memory=False)

quantile = 0.9
metadata = metadata[
    metadata['product_rating_count'] > metadata['product_rating_count'].quantile(quantile)].reset_index()
print(pd.DataFrame(metadata.columns, columns=['columns']).T)
print(metadata.head())

vectorizer = TfidfVectorizer(stop_words='english', max_df=0.8, min_df=2)
metadata.review_text = metadata.review_text.fillna('')
tfidf_model = vectorizer.fit_transform(metadata.review_text)
print(f'Matrix contains {tfidf_model.shape[0]} rows and {tfidf_model.shape[1]} columns')

popular_terms = ['cream', 'skin', 'awesome', 'happy', 'mascara', 'foundation', 'powder', 'blush']
columns = vectorizer.get_feature_names_out()
tdidf_df = pd.DataFrame.sparse.from_spmatrix(tfidf_model, columns=columns)
print(tdidf_df[popular_terms].head())


def get_content_based_recommendation(product_title, top_n=10, metric='cosine'):
    idx = metadata[metadata.product_title.str.lower() == product_title.lower()].empty

    model = NearestNeighbors(n_neighbors=top_n, metric=metric)
    model.fit(tfidf_model)
    similar_products = model.kneighbors(tfidf_model[idx], return_distance=False)[0]

    return metadata.iloc[similar_products]


print(get_content_based_recommendation('Olay Regenerist Whip Mini and Ultimate Eye Cream Combo')[
          ['product_title', 'review_rating', 'product_rating', 'product_rating_count', 'review_text']])
