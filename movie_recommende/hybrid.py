import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from content_based import content_based_filtering_enhanced
from collaborative import collaborative_filtering_enhanced
import warnings

warnings.filterwarnings('ignore')

# =====================================================================================
# == Functions for Streamlit App (main.py)
# =====================================================================================

def smart_hybrid_recommendation(merged_df, movie_title, genre_input, top_n=10, content_weight=0.5, collab_weight=0.5):
    """
    Generates hybrid recommendations by combining content-based and collaborative (KNN) methods.
    """

    # 1. Content-Based Recommendations
    content_recs = content_based_filtering_enhanced(
        merged_df,
        movie_title if movie_title else None,
        genre_input if genre_input else None,
        top_n=len(merged_df)
    )
    if content_recs is None or content_recs.empty:
        # Fallback to collaborative only if a movie title is provided
        if movie_title:
            return collaborative_filtering_enhanced(merged_df, movie_title, top_n)
        return content_recs  # May be None/empty if neither method can produce results

    content_recs['content_score'] = range(len(content_recs), 0, -1)

    # 2. Collaborative Recommendations (KNN)
    collab_recs = None
    if movie_title:
        collab_recs = collaborative_filtering_enhanced(merged_df, movie_title, top_n=len(merged_df))
    if collab_recs is None or collab_recs.empty:
        return content_recs.head(top_n)  # fallback to content-only

    collab_recs['collab_score'] = range(len(collab_recs), 0, -1)

    # 3. Merge Scores
    hybrid_df = pd.merge(content_recs, collab_recs[['Series_Title', 'collab_score']], on='Series_Title', how='left')
    hybrid_df['collab_score'].fillna(0, inplace=True)

    # Normalize scores
    scaler = MinMaxScaler()
    hybrid_df[['content_score', 'collab_score']] = scaler.fit_transform(
        hybrid_df[['content_score', 'collab_score']]
    )

    # Weighted Hybrid Score
    hybrid_df['hybrid_score'] = (hybrid_df['content_score'] * content_weight) + \
                                (hybrid_df['collab_score'] * collab_weight)

    # Remove the input movie itself
    if movie_title in hybrid_df['Series_Title'].values:
        hybrid_df = hybrid_df[hybrid_df['Series_Title'] != movie_title]

    return hybrid_df.sort_values(by='hybrid_score', ascending=False).head(top_n)

# =====================================================================================
# == Functions for Offline Evaluation (evaluate_recommendations.py)
# =====================================================================================

def predict_hybrid_ratings(user_id, movie_id, train_df, movies_df, tfidf_matrix, cosine_sim, indices, content_weight=0.5, collab_weight=0.5):
    """
    Predicts a single movie rating for a user by combining content and collaborative (KNN) methods.
    """

    # 1. Content-Based Prediction
    user_ratings = train_df[train_df['User_ID'] == user_id]
    content_pred_score = 3.0  # Default score
    if not user_ratings.empty and movie_id in indices:
        rated_movie_indices = [indices[mid] for mid in user_ratings['Movie_ID'] if mid in indices]
        target_movie_idx = indices[movie_id]
        sim_scores_to_rated = cosine_sim[target_movie_idx, rated_movie_indices]
        weighted_sim = np.dot(sim_scores_to_rated, user_ratings['Rating']) / user_ratings['Rating'].sum() \
                       if user_ratings['Rating'].sum() > 0 else 0
        content_pred_score = weighted_sim * 9 + 1  # scale 0–1 → 1–10

    # 2. Collaborative Prediction (fallback: neutral since KNN doesn’t predict rating easily)
    collab_pred_score = 5.0  

    # 3. Hybrid Weighted Score
    final_score = (content_pred_score * content_weight) + (collab_pred_score * collab_weight)
    return final_score
