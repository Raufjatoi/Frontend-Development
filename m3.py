import pandas as pd
import numpy as np

# Sample data (users x movies)
data = {
    'User': ['Mudassir', 'Huzaifa', 'Najaf', 'Awais'],
    'Inception': [5, 4, 0, 3],
    'Interstellar': [4, 5, 2, 0],
    'Titanic': [0, 5, 4, 0],
    'Toy Story': [0, 2, 5, 4]
}
df = pd.DataFrame(data).set_index('User')

def recommend(user):
    # Calculate similarities
    ratings = df.values
    norms = np.linalg.norm(ratings, axis=1, keepd=True)
    norms[norms == 0] = 1e-10  # Avoid division by zero
    norm_ratings = ratings / norms
    sim = norm_ratings @ norm_ratings.T
    
    # Find similar users (excluding self)
    user_idx = df.index.get_loc(user)
    similar_users = np.argsort(-sim[user_idx])[1:]  # - for descending order
    
    # Find unwatched movies
    target = df.loc[user]
    unwatched = target[target == 0].index
    
    # Calculate recommendations
    recs = {}
    for movie in unwatched:
        movie_idx = df.columns.get_loc(movie)
        ratings = df.iloc[similar_users, movie_idx]
        similarities = sim[user_idx, similar_users]
        
        # Weighted average (ignore zeros)
        mask = ratings > 0
        if mask.any():
            recs[movie] = (ratings[mask] * similarities[mask]).sum() / similarities[mask].sum()
    
    return sorted(recs.items(), key=lambda x: -x[1])[:2]

# Test it!
print("Recommendations for huzaifa:")
for movie, score in recommend('Huzaifa'):
    print(f"- {movie} (score: {score:.1f})")