import pandas as pd
import numpy as np
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from content_based import content_based_filtering_enhanced, create_content_features, find_rating_column, find_genre_column
from collaborative import collaborative_knn, load_user_ratings
import warnings
warnings.filterwarnings('ignore')


class LinearHybridRecommender:
    def __init__(self, merged_df):
        self.merged_df = merged_df
        self.rating_col = find_rating_column(merged_df)
        self.genre_col = find_genre_column(merged_df)
        self.user_ratings_df = load_user_ratings()
        # Weights
        self.alpha = 0.4  # Content
        self.beta = 0.3   # Collaborative
        self.gamma = 0.2  # Popularity
        self.delta = 0.1  # Recency

    def _content_scores(self, target_movie, genre, top_n):
        scores = {}
        results = content_based_filtering_enhanced(self.merged_df, target_movie, genre, top_n * 3)
        if results is not None and not results.empty:
            max_rating = results[self.rating_col].max()
            if max_rating and max_rating > 0:
                for _, row in results.iterrows():
                    scores[row['Series_Title']] = float(row[self.rating_col]) / float(max_rating)
        return scores

    def _collab_scores(self, target_movie, top_n):
        scores = {}
        if target_movie and self.user_ratings_df is not None:
            results = collaborative_knn(self.merged_df, target_movie, top_n=top_n * 3)
            if results is not None and not results.empty:
                max_rating = results[self.rating_col].max() if self.rating_col in results.columns else None
                for _, row in results.iterrows():
                    if max_rating and max_rating > 0 and self.rating_col in results.columns:
                        scores[row['Series_Title']] = float(row[self.rating_col]) / float(max_rating)
                    else:
                        # If no rating col, treat presence as score 1.0
                        scores[row['Series_Title']] = 1.0
        return scores

    def _popularity_scores(self):
        pop = {}
        votes_col = 'No_of_Votes' if 'No_of_Votes' in self.merged_df.columns else 'Votes'
        for _, movie in self.merged_df.iterrows():
            title = movie['Series_Title']
            rating = movie.get(self.rating_col, 7.0)
            votes = movie.get(votes_col, 1000)
            try:
                votes_val = float(str(votes).replace(',', ''))
            except Exception:
                votes_val = 1000.0
            if pd.isna(rating):
                rating = 7.0
            popularity = (float(rating) * np.log10(votes_val + 1.0)) / 10.0
            pop[title] = float(np.clip(popularity, 0.0, 1.0))
        # Optional: light boost based on user interactions if available
        if self.user_ratings_df is not None and 'Movie_ID' in self.merged_df.columns:
            interaction_counts = self.user_ratings_df['Movie_ID'].value_counts()
            for mid, cnt in interaction_counts.items():
                match = self.merged_df[self.merged_df['Movie_ID'] == mid]
                if not match.empty:
                    t = match.iloc[0]['Series_Title']
                    boost = min(cnt / 100.0, 1.0)
                    pop[t] = 0.6 * pop.get(t, 0.5) + 0.4 * boost
        return pop

    def _recency_scores(self):
        rec = {}
        year_col = 'Released_Year' if 'Released_Year' in self.merged_df.columns else 'Year'
        current_year = pd.Timestamp.now().year
        for _, movie in self.merged_df.iterrows():
            title = movie['Series_Title']
            year = movie.get(year_col, 2000)
            try:
                year_val = int(str(year).split()[0]) if not pd.isna(year) else 2000
            except Exception:
                year_val = 2000
            diff = max(0, current_year - year_val)
            recency = np.exp(-diff / 20.0)
            rec[title] = float(np.clip(recency, 0.0, 1.0))
        return rec

    def recommend(self, target_movie=None, genre=None, top_n=8):
        # Gather component scores
        content_scores = self._content_scores(target_movie, genre, top_n)
        collab_scores = self._collab_scores(target_movie, top_n)
        popularity_scores = self._popularity_scores()
        recency_scores = self._recency_scores()

        # Candidate set
        candidates = set(content_scores.keys()) | set(collab_scores.keys())
        if len(candidates) < top_n * 2:
            # Add top popular titles
            for t, _ in sorted(popularity_scores.items(), key=lambda x: x[1], reverse=True)[:top_n * 2]:
                candidates.add(t)

        final_scores = {}
        for title in candidates:
            c = content_scores.get(title, 0.0)
            cf = collab_scores.get(title, 0.0)
            pop = popularity_scores.get(title, 0.5)
            rec = recency_scores.get(title, 0.5)
            score = self.alpha * c + self.beta * cf + self.gamma * pop + self.delta * rec
            final_scores[title] = float(score)

        top_items = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        top_titles = [t for t, _ in top_items]
        result_df = self.merged_df[self.merged_df['Series_Title'].isin(top_titles)]
        if result_df.empty:
            return None
        # Preserve order
        order = {t: i for i, t in enumerate(top_titles)}
        result_df = result_df.copy()
        result_df['rank_order'] = result_df['Series_Title'].map(order)
        result_df = result_df.sort_values('rank_order').drop(columns=['rank_order'])
        return result_df[['Series_Title', self.genre_col, self.rating_col]]


@st.cache_data
def smart_hybrid_recommendation(merged_df, target_movie=None, genre=None, top_n=8):
    recommender = LinearHybridRecommender(merged_df)
    return recommender.recommend(target_movie, genre, top_n)