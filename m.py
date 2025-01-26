import numpy as n
from sklearn.metrics.pairwise import cosine_similarity

movies = ["Inception", "Interstellar", "Titanic", "The Dark Knight", "Toy Story"]
users = {
    "Mudassir": [5, 4, 0, 3, 0],   # 0 = hasn't watched
    "Huzaifa": [4, 5, 5, 5, 2],
    "Ahsan": [3, 2, 4, 4, 5],
    "Umar": [5, 5, 0, 4, 1],
    "Rehman": [0, 3, 4, 0, 5]
}

def get_recommendations(target_user, users, movies, n=3):
    # Convert ratings to matrix
    user_names = list(users.keys())
    ratings = n.array(list(users.values()))
    
    # Calculate similarity scores
    similarity = cosine_similarity(ratings)
    
    # Find target user's index
    target_idx = user_names.index(target_user)
    
    # Get most similar users (excluding self)
    similar_users = n.argsort(similarity[target_idx])[::-1][1:]
    
    # Find movies target hasn't watched
    target_ratings = ratings[target_idx]
    unwatched = n.where(target_ratings == 0)[0]
    
    # Calculate recommendation scores
    recommendations = {}
    for movie_idx in unwatched:
        # Get ratings from similar users
        similar_ratings = ratings[similar_users, movie_idx]
        
        # Calculate weighted average (using similarity as weight)
        total_score = 0
        total_weight = 0
        for i, rating in enumerate(similar_ratings):
            if rating > 0:  # Only consider users who watched it
                weight = similarity[target_idx, similar_users[i]]
                total_score += rating * weight
                total_weight += weight
        
        if total_weight > 0:
            recommendations[movies[movie_idx]] = total_score / total_weight
    
    # Return top recommendations
    return sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:n]

target_user = "Rehman"
recommendations = get_recommendations(target_user, users, movies)

print(f"\nRecommendations for {target_user}:")
for movie, score in recommendations:
    print(f"- {movie} (score: {score:.2f})")

print("\nHow it works:")
print(f"1. Finds users with similar taste to {target_user}")
print("2. Checks which movies they liked that you haven't watched")
print("3. Suggests highest-rated movies from similar users!")
