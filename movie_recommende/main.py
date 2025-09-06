import streamlit as st
import pandas as pd
import numpy as np
import warnings
import requests
import io

warnings.filterwarnings('ignore')

# Import functions with error handling
try:
    from content_based import content_based_filtering_enhanced
except ImportError:
    content_based_filtering_enhanced = None

try:
    from collaborative import collaborative_filtering_enhanced
except ImportError:
    collaborative_filtering_enhanced = None
    
try:
    from hybrid import smart_hybrid_recommendation
except ImportError:
    smart_hybrid_recommendation = None

# Import column utilities
try:
    from column_utils import (
        get_genre_column, get_overview_column, get_rating_column, 
        get_year_column, get_votes_column, safe_get_column_data,
        apply_genre_filter, get_movie_display_info
    )
except ImportError:
    # Fallback functions if column_utils.py is not available
    def get_genre_column(df):
        for col in ['Genre', 'Genre_y', 'Genre_x', 'Genres']:
            if col in df.columns:
                return col
        return None
    
    def get_rating_column(df):
        for col in ['IMDB_Rating', 'Rating', 'IMDB_Rating_y', 'IMDB_Rating_x']:
            if col in df.columns:
                return col
        return None
    
    def get_year_column(df):
        for col in ['Released_Year', 'Year', 'Released_Year_y', 'Released_Year_x']:
            if col in df.columns:
                return col
        return None
    
    def apply_genre_filter(df, genre_filter):
        genre_col = get_genre_column(df)
        if genre_col:
            return df[df[genre_col].str.contains(genre_filter, case=False, na=False)]
        else:
            return pd.DataFrame()
    
    def get_movie_display_info(df, movie_row):
        rating_col = get_rating_column(df)
        genre_col = get_genre_column(df)
        year_col = get_year_column(df)
        
        return {
            'title': movie_row.get('Series_Title', 'Unknown'),
            'rating': movie_row.get(rating_col, 'N/A') if rating_col else 'N/A',
            'genre': movie_row.get(genre_col, 'N/A') if genre_col else 'N/A',
            'year': movie_row.get(year_col, 'N/A') if year_col else 'N/A',
            'poster': movie_row.get('Poster_Link', '')
        }

# Backup content-based function
def simple_content_based(merged_df, target_movie, genre_filter=None, top_n=10):
    """Simplified content-based filtering using available columns with proper column resolution"""
    if not target_movie and not genre_filter:
        return pd.DataFrame()
    
    # Handle genre-only filtering
    if genre_filter and not target_movie:
        filtered = apply_genre_filter(merged_df, genre_filter)
        if not filtered.empty:
            rating_col = get_rating_column(filtered)
            if rating_col:
                filtered = filtered.sort_values(rating_col, ascending=False)
            return filtered.head(top_n)
        return pd.DataFrame()
    
    # Movie-based filtering
    if target_movie not in merged_df['Series_Title'].values:
        return pd.DataFrame()
    
    # Simple genre-based similarity
    target_row = merged_df[merged_df['Series_Title'] == target_movie].iloc[0]
    genre_col = get_genre_column(merged_df)
    
    if not genre_col:
        return pd.DataFrame()
    
    target_genres = str(target_row[genre_col]).split(', ') if pd.notna(target_row[genre_col]) else []
    
    # Find movies with similar genres
    similar_movies = []
    for idx, row in merged_df.iterrows():
        if row['Series_Title'] == target_movie:
            continue
        
        movie_genres = str(row[genre_col]).split(', ') if pd.notna(row[genre_col]) else []
        common_genres = set(target_genres) & set(movie_genres)
        
        if common_genres:
            similar_movies.append((idx, len(common_genres)))
    
    # Sort by genre similarity and rating
    similar_movies.sort(key=lambda x: x[1], reverse=True)
    top_indices = [x[0] for x in similar_movies[:top_n*2]]
    
    results = merged_df.loc[top_indices]
    
    # Apply genre filter if provided
    if genre_filter:
        results = apply_genre_filter(results, genre_filter)
    
    # Sort by rating
    rating_col = get_rating_column(results)
    if rating_col:
        results = results.sort_values(rating_col, ascending=False)
    
    return results.head(top_n)

# =========================
# Streamlit Configuration
# =========================
st.set_page_config(
    page_title="🎬 Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🎬 Movie Recommendation System")
st.markdown("---")

# =========================
# GitHub CSV Loading Functions
# =========================
@st.cache_data
def load_csv_from_github(file_url, file_name):
    """Load CSV file from GitHub repository - silent version"""
    try:
        response = requests.get(file_url, timeout=30)
        response.raise_for_status()
        
        # Read CSV from response content
        csv_content = io.StringIO(response.text)
        df = pd.read_csv(csv_content)
        
        # Silent success - no st.success message
        return df
        
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Failed to load {file_name} from GitHub: {str(e)}")
        return None
    except pd.errors.EmptyDataError:
        st.error(f"❌ {file_name} is empty or corrupted")
        return None
    except Exception as e:
        st.error(f"❌ Error processing {file_name}: {str(e)}")
        return None

@st.cache_data
def load_and_prepare_data():
    """Load CSVs from GitHub and prepare data for recommendation algorithms - silent version"""
    
    # GitHub raw file URLs - replace with your actual repository URLs
    github_base_url = "https://raw.githubusercontent.com/yy9449/recommender/main/movie_recommende/"
    
    # File URLs
    movies_url = github_base_url + "movies.csv"
    imdb_url = github_base_url + "imdb_top_1000.csv"
    user_ratings_url = github_base_url + "user_movie_rating.csv"
    
    # Silent loading - show minimal progress info
    with st.spinner("Loading datasets..."):
        movies_df = load_csv_from_github(movies_url, "movies.csv")
        imdb_df = load_csv_from_github(imdb_url, "imdb_top_1000.csv")
        user_ratings_df = load_csv_from_github(user_ratings_url, "user_movie_rating.csv")
    
    # Check if required files loaded successfully
    if movies_df is None or imdb_df is None:
        return None, None, "❌ Required CSV files (movies.csv, imdb_top_1000.csv) could not be loaded from GitHub"
    
    # Store user ratings in session state for other functions to access - silent
    if user_ratings_df is not None:
        st.session_state['user_ratings_df'] = user_ratings_df
        # Silent success - no message
    else:
        # Only show warning if explicitly needed
        if 'user_ratings_df' in st.session_state:
            del st.session_state['user_ratings_df']
    
    try:
        # Validate required columns
        if 'Series_Title' not in movies_df.columns or 'Series_Title' not in imdb_df.columns:
            return None, None, "❌ Missing Series_Title column in one or both datasets"
        
        # Check if movies.csv has Movie_ID
        if 'Movie_ID' not in movies_df.columns:
            movies_df['Movie_ID'] = range(len(movies_df))
            # Silent addition - no info message
        
        # Merge on Series_Title
        merged_df = pd.merge(movies_df, imdb_df, on="Series_Title", how="inner")
        merged_df = merged_df.drop_duplicates(subset="Series_Title")
        
        # Ensure Movie_ID is preserved in merged dataset
        if 'Movie_ID' not in merged_df.columns and 'Movie_ID' in movies_df.columns:
            # Re-merge to preserve Movie_ID
            merged_df = pd.merge(movies_df[['Movie_ID', 'Series_Title']], merged_df, on="Series_Title", how="inner")
        
        # Silent success - no success message
        return merged_df, user_ratings_df, None
        
    except Exception as e:
        return None, None, f"❌ Error merging datasets: {str(e)}"

# Alternative: Try local files if GitHub fails
@st.cache_data
def load_local_fallback():
    """Fallback to load local files if GitHub loading fails - silent version"""
    try:
        import os
        
        # Try different possible file paths
        movies_df = None
        imdb_df = None
        user_ratings_df = None
        
        # Check for movies.csv
        for path in ["movies.csv", "./movies.csv", "data/movies.csv", "../movies.csv"]:
            if os.path.exists(path):
                movies_df = pd.read_csv(path)
                break
        
        # Check for imdb_top_1000.csv
        for path in ["imdb_top_1000.csv", "./imdb_top_1000.csv", "data/imdb_top_1000.csv", "../imdb_top_1000.csv"]:
            if os.path.exists(path):
                imdb_df = pd.read_csv(path)
                break
        
        # Check for user_movie_rating.csv
        for path in ["user_movie_rating.csv", "./user_movie_rating.csv", "data/user_movie_rating.csv", "../user_movie_rating.csv"]:
            if os.path.exists(path):
                user_ratings_df = pd.read_csv(path)
                break
        
        if movies_df is None or imdb_df is None:
            return None, None, "Required CSV files not found locally either"
        
        # Store user ratings in session state - silent
        if user_ratings_df is not None:
            st.session_state['user_ratings_df'] = user_ratings_df
        
        # Check if movies.csv has Movie_ID
        if 'Movie_ID' not in movies_df.columns:
            movies_df['Movie_ID'] = range(len(movies_df))
        
        # Merge on Series_Title
        merged_df = pd.merge(movies_df, imdb_df, on="Series_Title", how="inner")
        merged_df = merged_df.drop_duplicates(subset="Series_Title")
        
        # Ensure Movie_ID is preserved in merged dataset
        if 'Movie_ID' not in merged_df.columns and 'Movie_ID' in movies_df.columns:
            merged_df = pd.merge(movies_df[['Movie_ID', 'Series_Title']], merged_df, on="Series_Title", how="inner")
        
        return merged_df, user_ratings_df, None
        
    except Exception as e:
        return None, None, str(e)

def display_movie_posters(results_df, merged_df):
    """Display movie posters in cinema-style layout (5 columns per row) with proper column resolution"""
    if results_df is None or results_df.empty:
        return
    
    # Get movies with posters using proper column resolution
    movies_with_posters = []
    for _, row in results_df.iterrows():
        movie_title = row['Series_Title']
        full_movie_info = merged_df[merged_df['Series_Title'] == movie_title].iloc[0]
        
        # Use column utilities for consistent data extraction
        movie_info = get_movie_display_info(merged_df, full_movie_info)
        movies_with_posters.append(movie_info)
    
    # Display in rows of 5 columns
    movies_per_row = 5
    
    for i in range(0, len(movies_with_posters), movies_per_row):
        cols = st.columns(movies_per_row)
        row_movies = movies_with_posters[i:i + movies_per_row]
        
        for j, movie in enumerate(row_movies):
            with cols[j]:
                # Movie poster with consistent sizing
                poster_url = movie['poster']
                if poster_url and pd.notna(poster_url) and poster_url.strip():
                    try:
                        st.image(
                            poster_url, 
                            width=200  # Fixed width for consistency
                        )
                    except:
                        # Fallback if image fails to load
                        st.container()
                        st.markdown(
                            f"""
                            <div style='
                                width: 200px; 
                                height: 300px; 
                                background-color: #f0f0f0; 
                                display: flex; 
                                align-items: center; 
                                justify-content: center;
                                border: 1px solid #ddd;
                                border-radius: 8px;
                            '>
                                <p style='text-align: center; color: #666;'>🎬<br>No Image<br>Available</p>
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )
                else:
                    # No poster available - show placeholder
                    st.markdown(
                        f"""
                        <div style='
                            width: 200px; 
                            height: 300px; 
                            background-color: #f0f0f0; 
                            display: flex; 
                            align-items: center; 
                            justify-content: center;
                            border: 1px solid #ddd;
                            border-radius: 8px;
                            margin-bottom: 10px;
                        '>
                            <p style='text-align: center; color: #666;'>🎬<br>No Image<br>Available</p>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                
                # Movie information below poster
                st.markdown(f"**{movie['title'][:25]}{'...' if len(movie['title']) > 25 else ''}**")
                st.markdown(f"⭐ {movie['rating']}/10")
                st.markdown(f"📅 {movie['year']}")
                
                # Genre with text wrapping
                genre_text = str(movie['genre'])[:30] + "..." if len(str(movie['genre'])) > 30 else str(movie['genre'])
                st.markdown(f"🎭 {genre_text}")
                
                # Add some spacing between movies
                st.markdown("---")

# =========================
# Main Application
# =========================
def main():
    # Load data from GitHub repository first, then fallback to local
    merged_df, user_ratings_df, error = load_and_prepare_data()
    
    # If GitHub loading failed, try local fallback
    if merged_df is None:
        st.warning("⚠️ GitHub loading failed, trying local files...")
        merged_df, user_ratings_df, local_error = load_local_fallback()
        
        if merged_df is None:
            st.error("❌ Could not load datasets from GitHub or local files.")
            
            # Show detailed error info
            with st.expander("🔍 Error Details"):
                st.write("**GitHub Error:**", error if error else "Unknown error")
                st.write("**Local Error:**", local_error if local_error else "Unknown error")
            
            st.info("""
            **Setup Instructions:**
            
            **For GitHub Loading (Recommended):**
            1. Update the GitHub URLs in the code with your actual repository details
            2. Make sure your CSV files are in the main branch
            3. Ensure the repository is public or accessible
            
            **Required Files:**
            - `movies.csv`: Movie metadata with Movie_ID and Series_Title columns
            - `imdb_top_1000.csv`: IMDB movie data with ratings and genres  
            - `user_movie_rating.csv`: Optional user ratings file
            
            **GitHub URL Format:**
            ```
            https://raw.githubusercontent.com/YOUR_USERNAME/YOUR_REPO_NAME/main/FILENAME.csv
            ```
            """)
            st.stop()
    
    # Show minimal success message only
    st.success("🎉 Ready to recommend!")
    
    # Show data summary
    with st.expander("📊 Dataset Summary", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Movies", len(merged_df))
        
        with col2:
            if user_ratings_df is not None:
                st.metric("User Ratings", len(user_ratings_df))
            else:
                st.metric("User Data", "Synthetic")
        
        with col3:
            if user_ratings_df is not None:
                st.metric("Unique Users", user_ratings_df['User_ID'].nunique())
            else:
                st.metric("Algorithm Mode", "Enhanced")

    # Silent check for user ratings availability
    user_ratings_available = user_ratings_df is not None

    # Sidebar
    st.sidebar.header("🎯 Recommendation Settings")
    
    # New input method - can select both movie and genre
    st.sidebar.subheader("🔍 Input Selection")
    
    # Movie selection
    st.sidebar.markdown("**🎬 Movie Selection**")
    all_movie_titles = sorted(merged_df['Series_Title'].dropna().unique().tolist())
    movie_title = st.sidebar.selectbox(
        "Select a Movie (Optional):",
        options=[""] + all_movie_titles,
        index=0,
        help="Choose a movie to get similar recommendations"
    )
    
    # Genre selection with proper column resolution
    st.sidebar.markdown("**🎭 Genre Selection**")
    genre_col = get_genre_column(merged_df)
    
    if genre_col:
        all_genres = set()
        for genre_str in merged_df[genre_col].dropna():
            if isinstance(genre_str, str):
                all_genres.update([g.strip() for g in genre_str.split(',')])
        
        sorted_genres = sorted(all_genres)
        genre_input = st.sidebar.selectbox(
            "Select Genre (Optional):", 
            options=[""] + sorted_genres,
            help="Choose a genre to filter recommendations"
        )
    else:
        st.sidebar.error("❌ No genre column found in dataset")
        genre_input = ""
    
    # Show input combination info
    if movie_title and genre_input:
        st.sidebar.success("🎯 Using both movie and genre for enhanced recommendations!")
    elif movie_title:
        st.sidebar.info("🎬 Using movie-based recommendations")
    elif genre_input:
        st.sidebar.info("🎭 Using genre-based recommendations")
    else:
        st.sidebar.warning("⚠️ Please select at least a movie or genre")
    
    # Show selected movie info if movie is selected
    if movie_title:
        movie_info = merged_df[merged_df['Series_Title'] == movie_title].iloc[0]
        
        with st.sidebar.expander("ℹ️ Selected Movie Info", expanded=True):
            # Use proper column resolution for display
            display_info = get_movie_display_info(merged_df, movie_info)
            
            st.write(f"**🎬 {movie_title}**")
            if 'Movie_ID' in movie_info.index:
                st.write(f"**🆔 Movie ID:** {movie_info['Movie_ID']}")
            st.write(f"**🎭 Genre:** {display_info['genre']}")
            st.write(f"**⭐ Rating:** {display_info['rating']}/10")
            st.write(f"**📅 Year:** {display_info['year']}")
    
    # Algorithm selection
    algorithm = st.sidebar.selectbox(
        "🔬 Choose Algorithm:",
        ["Hybrid", "Content-Based", "Collaborative Filtering"]
    )
    
    # Number of recommendations
    top_n = st.sidebar.slider("📊 Number of Recommendations:", 3, 15, 8)
    
    # Show data source info quietly in sidebar
    if user_ratings_available:
        st.sidebar.success("💾 Real user data available")
    else:
        st.sidebar.info("🤖 Using synthetic profiles")
    
    # Generate button
    if st.sidebar.button("🚀 Generate Recommendations", type="primary"):
        if not movie_title and not genre_input:
            st.error("❌ Please provide either a movie title or select a genre!")
            return
        
        with st.spinner("🎬 Generating personalized recommendations..."):
            results = None
            
            try:
                if algorithm == "Content-Based":
                    # Try original function first, fallback to simple version
                    if content_based_filtering_enhanced is not None:
                        try:
                            results = content_based_filtering_enhanced(
                                merged_df, 
                                target_movie=movie_title if movie_title else None,
                                genre_filter=genre_input if genre_input else None,
                                top_n=top_n
                            )
                        except:
                            # Fallback to simple function
                            results = simple_content_based(
                                merged_df, 
                                target_movie=movie_title if movie_title else None,
                                genre_filter=genre_input if genre_input else None,
                                top_n=top_n
                            )
                    else:
                        results = simple_content_based(
                            merged_df, 
                            target_movie=movie_title if movie_title else None,
                            genre_filter=genre_input if genre_input else None,
                            top_n=top_n
                        )
                    
                elif algorithm == "Collaborative Filtering":
                    if movie_title and user_ratings_df is not None and collaborative_filtering_enhanced is not None:
                        results = collaborative_filtering_enhanced(merged_df, user_ratings_df, movie_title, top_n)
                    else:
                        st.warning("Collaborative filtering requires a movie title and user ratings data.")
                        return
                        
                else:  # Hybrid
                    if user_ratings_df is not None and smart_hybrid_recommendation is not None:
                        try:
                            results = smart_hybrid_recommendation(
                                merged_df,
                                user_ratings_df=user_ratings_df,
                                target_movie=movie_title if movie_title else None,
                                genre_filter=genre_input if genre_input else None,
                                top_n=top_n
                            )
                        except Exception as hybrid_error:
                            # Fallback to content-based
                            st.error(f"Hybrid filtering failed with error: {str(hybrid_error)}")
                            st.info("Falling back to content-based recommendations.")
                            results = simple_content_based(
                                merged_df,
                                target_movie=movie_title if movie_title else None,
                                genre_filter=genre_input if genre_input else None,
                                top_n=top_n
                            )
                    else:
                        # Fallback to content-based if no user data
                        st.info("No user data available, falling back to content-based recommendations.")
                        results = simple_content_based(
                            merged_df,
                            target_movie=movie_title if movie_title else None,
                            genre_filter=genre_input if genre_input else None,
                            top_n=top_n
                        )
                        
            except Exception as e:
                st.error(f"❌ Error generating recommendations: {str(e)}")
                st.info("💡 Try using different parameters or check your data format.")
                return
            
            # Display results
            if results is not None and not results.empty:
                # Results display
                st.subheader("🎬 Recommended Movies")
                
                # Cinema-style poster display
                display_movie_posters(results, merged_df)
                
                # Optional: Show detailed table with proper column names
                with st.expander("📊 View Detailed Information", expanded=False):
                    # Format the results for better display using column utilities
                    display_results = results.copy()
                    rating_col = get_rating_column(results)
                    genre_col = get_genre_column(results)
                    
                    # Rename columns for display
                    rename_dict = {'Series_Title': 'Movie Title'}
                    if genre_col:
                        rename_dict[genre_col] = 'Genre'
                    if rating_col:
                        rename_dict[rating_col] = 'IMDB Rating'
                    
                    display_results = display_results.rename(columns=rename_dict)
                    
                    # Add ranking
                    display_results.insert(0, 'Rank', range(1, len(display_results) + 1))
                    
                    # Add Movie_ID if available
                    if 'Movie_ID' in merged_df.columns:
                        movie_ids = []
                        for _, row in results.iterrows():
                            movie_info = merged_df[merged_df['Series_Title'] == row['Series_Title']]
                            if not movie_info.empty:
                                movie_ids.append(movie_info.iloc[0]['Movie_ID'])
                            else:
                                movie_ids.append('N/A')
                        display_results.insert(1, 'Movie ID', movie_ids)
                    
                    st.dataframe(
                        display_results,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Rank": st.column_config.NumberColumn("Rank", width="small"),
                            "Movie ID": st.column_config.NumberColumn("Movie ID", width="small"),
                            "Movie Title": st.column_config.TextColumn("Movie Title", width="large"),
                            "Genre": st.column_config.TextColumn("Genre", width="medium"),
                            "IMDB Rating": st.column_config.NumberColumn("IMDB Rating", format="%.1f⭐")
                        }
                    )
                
                # Enhanced insights with proper column resolution
                st.subheader("📈 Recommendation Insights")
                
                # Get proper column names
                rating_col = get_rating_column(results)
                genre_col = get_genre_column(results)
                
                if rating_col and genre_col:
                    # Create columns for metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        avg_rating = results[rating_col].mean()
                        st.metric("Average Rating", f"{avg_rating:.1f}⭐")
                    
                    with col2:
                        total_movies = len(results)
                        st.metric("Total Recommendations", total_movies)
                    
                    with col3:
                        # Highest rated movie
                        max_rating = results[rating_col].max()
                        st.metric("Highest Rating", f"{max_rating:.1f}⭐")
                    
                    with col4:
                        # Most common genre
                        genres_list = []
                        for genre_str in results[genre_col].dropna():
                            genres_list.extend([g.strip() for g in str(genre_str).split(',')])
                        
                        if genres_list:
                            most_common_genre = pd.Series(genres_list).mode().iloc[0] if len(pd.Series(genres_list).mode()) > 0 else "Various"
                            st.metric("Top Genre", most_common_genre)
                    
                    # Genre and rating distribution
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if genres_list:
                            st.subheader("🎭 Genre Distribution")
                            genre_counts = pd.Series(genres_list).value_counts().head(8)
                            st.bar_chart(genre_counts)
                    
                    with col2:
                        st.subheader("⭐ Rating Distribution")
                        rating_bins = pd.cut(results[rating_col], bins=5, labels=['Low', 'Below Avg', 'Average', 'Above Avg', 'High'])
                        rating_dist = rating_bins.value_counts()
                        st.bar_chart(rating_dist)
                    
                    # Show input combination effect if both were used
                    if movie_title and genre_input:
                        st.subheader("🎯 Input Combination Analysis")
                        
                        # Show genre matching in results
                        genre_matches = 0
                        for _, row in results.iterrows():
                            if genre_input.lower() in str(row[genre_col]).lower():
                                genre_matches += 1
                        
                        match_percentage = (genre_matches / len(results)) * 100
                        st.info(f"📊 {genre_matches}/{len(results)} recommendations ({match_percentage:.1f}%) match your selected genre '{genre_input}'")
            
            else:
                st.error("❌ No recommendations found. Try different inputs or algorithms.")
                
                # Provide suggestions
                st.subheader("💡 Suggestions:")
                if movie_title and not genre_input:
                    st.write("- Try adding a genre preference")
                    st.write("- Try a different algorithm (Content-Based might work better)")
                elif genre_input and not movie_title:
                    st.write("- Try selecting a movie you like")
                    st.write("- Try a more common genre")
                else:
                    st.write("- Check if the movie title is spelled correctly")
                    st.write("- Try selecting from the dropdown instead of typing")

if __name__ == "__main__":
    main()
