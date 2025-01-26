import numpy as np
import pandas as pd

data = {
    "User": ["Mudassir", "Ahsan", "Umar", "Awais", "Huzaifa"],
    "Inception": [5, 4, 3, 5, 0],
    "Interstellar": [4, 5, 2, 5, 3],
    "Titanic": [0, 5, 4, 0, 4],
    "The Dark Knight": [3, 5, 4, 4, 0],
    "Toy Story": [0, 2, 5, 1, 5]
}

df = pd.DataFrame(data).set_index("User")

def cosine_similarity(user1, user2):
    # Get common rated movies
    mask = (user1 > 0) & (user2 > 0)
    a = user1[mask]
    b = user2[mask]
    
    if len(a) == 0:
        return 0  # No common ratings
    
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    return dot_product / (norm_a * norm_b)

def recommend(user, n=3):
    # Get target user's ratings
    target = df.loc[user]
    
    # Calculate similarity with all users
    similarities = {}
    for other_user in df.index:
        if other_user != user:
            sim = cosine_similarity(target, df.loc[other_user])
            similarities[other_user] = sim
    
    # Get top similar users
    similar_users = sorted(similarities.items(), 
                          key=lambda x: x[1], 
                          reverse=True)[:n]
    
    # Find movies target hasn't watched
    unwatched = target[target == 0].index
    
    # Calculate recommendation scores
    recommendations = {}
    for movie in unwatched:
        weighted_sum = 0
        similarity_sum = 0
        
        for other_user, similarity in similar_users:
            rating = df.loc[other_user, movie]
            if rating > 0:  # Only consider users who watched it
                weighted_sum += rating * similarity
                similarity_sum += similarity
                
        if similarity_sum > 0:
            recommendations[movie] = weighted_sum / similarity_sum
    
    return sorted(recommendations.items(), 
                 key=lambda x: x[1], 
                 reverse=True)

print("Recommendations for Huzaifa:")
for movie, score in recommend("Huzaifa"):
    print(f"- {movie} (score: {score:.2f})")