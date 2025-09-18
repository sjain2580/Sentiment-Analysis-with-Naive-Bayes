# Step 1: Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
import string
from collections import Counter

# Step 2: Load a small, built-in dataset
# In a real-world scenario, you would load a large dataset like the IMDB reviews.
# For demonstration purposes, we use a small, hard-coded dataset.
reviews = [
    "This movie was fantastic and brilliant! The acting was superb.",
    "A terrible movie with a weak plot and poor direction. A disappointment.",
    "The plot was good, but the acting was a little slow and uninspired.",
    "An absolute masterpiece of cinema. I loved every minute of it.",
    "The film was boring and not entertaining. I would not recommend it.",
    "Great movie! The climax was so thrilling and exciting.",
    "Boring and a waste of time. I did not like it.",
    "A truly wonderful experience. I would see it again.",
    "Very bad acting and the story was confusing. A bad film.",
    "The most enjoyable film I have seen in years. Highly recommend!"
]

labels = ['positive', 'negative', 'negative', 'positive', 'negative', 'positive', 'negative', 'positive', 'negative', 'positive']

print("Dataset loaded.")

# Step 3: Preprocess the text
# This function converts text to lowercase and removes punctuation.
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

preprocessed_reviews = [preprocess_text(review) for review in reviews]

print("\nText preprocessing complete.")

# Step 4: Convert text to numerical data using TfidfVectorizer (Bag-of-Words)
# We will use TfidfVectorizer, a more advanced form of Bag-of-Words, in our pipeline.
# This converts text documents to a matrix of TF-IDF features.

# Step 5: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(preprocessed_reviews, labels, test_size=0.2, random_state=42)

print("\nData split into training and testing sets.")
print(f"Training set size: {len(X_train)} reviews")
print(f"Testing set size: {len(X_test)} reviews")

# Step 6: Hyperparameter Tuning with Pipeline and GridSearchCV
# We use a pipeline to chain the vectorizer and the classifier, which is
# best practice for machine learning workflows.
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('classifier', MultinomialNB())
])

# Define the hyperparameter grid for tuning 'alpha'
param_grid = {
    'classifier__alpha': np.linspace(0.1, 1.0, 10)
}

# Use GridSearchCV to find the best 'alpha'
print("\nPerforming GridSearchCV to find the best 'alpha' parameter. This may take a moment...")
grid_search = GridSearchCV(pipeline, param_grid, cv=3, verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
best_alpha = grid_search.best_params_['classifier__alpha']

print("\n--- Hyperparameter Tuning Results ---")
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
print(f"Best 'alpha' parameter found: {best_alpha:.2f}")
print("-------------------------------------")

# Step 7: Make predictions and evaluate the best model
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\n--- Final Model Evaluation ---")
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("----------------------------")

# Step 8: Visualization
# Function to generate a word cloud
def generate_word_cloud(text, title):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(title)
    plt.axis('off')
    plt.savefig('cloudword1.png')
    plt.show()

# Separate positive and negative reviews
positive_reviews = " ".join([review for review, label in zip(preprocessed_reviews, labels) if label == 'positive'])
negative_reviews = " ".join([review for review, label in zip(preprocessed_reviews, labels) if label == 'negative'])

# Word clouds for each sentiment class
print("\nGenerating Word Clouds...")
generate_word_cloud(positive_reviews, 'Most Frequent Words in Positive Reviews')
generate_word_cloud(negative_reviews, 'Most Frequent Words in Negative Reviews')

# Distribution Plot of word counts
all_words = positive_reviews.split() + negative_reviews.split()
word_counts = Counter(all_words)
most_common_words = word_counts.most_common(20)

words = [word for word, count in most_common_words]
counts = [count for word, count in most_common_words]

plt.figure(figsize=(12, 6))
plt.bar(words, counts)
plt.title('Distribution of 20 Most Frequent Words')
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('plot.png')
plt.show()
