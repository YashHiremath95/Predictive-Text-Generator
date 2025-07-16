import random
import os
from collections import defaultdict
from flask import Flask, request, render_template

class PredictiveTextGenerator:
    def __init__(self, n=3):  # Changed default n to 3 for higher-order n-grams
        """
        Initialize the text generator with Markov chain order n.
        n determines how many previous words to consider for prediction.
        """
        self.n = n
        self.chain = defaultdict(list)
        self.start_words = []
        self.word_counts = defaultdict(int)  # Track word frequencies
        self.case_sensitive = False  # Add a flag for case sensitivity

    def train(self, text):
        """
        Train the model on the input text
        """
        if not self.case_sensitive:
            text = text.lower()  # Convert text to lowercase if case sensitivity is disabled
        words = text.split()

        # Not enough words to build the model
        if len(words) <= self.n:
            return

        self.start_words.append(tuple(words[:self.n]))

        for i in range(len(words) - self.n):
            current_state = tuple(words[i:i+self.n])
            next_word = words[i+self.n]
            self.chain[current_state].append(next_word)
            self.word_counts[next_word] += 1  # Track word frequency

    def generate(self, length=50, seed_text=None):
        """
        Generate new text of specified length
        """
        if not self.chain:
            return "Model not trained yet. Please provide training text first."

        if seed_text and not self.case_sensitive:
            seed_text = seed_text.lower()  # Convert seed text to lowercase if case sensitivity is disabled

        # Start with a seed or random start word
        if seed_text:
            seed_words = seed_text.split()
            if len(seed_words) >= self.n:
                current_state = tuple(seed_words[-self.n:])
            else:
                current_state = random.choice(self.start_words)
        else:
            current_state = random.choice(self.start_words)

        generated_words = list(current_state)

        for _ in range(length):
            possible_words = self.chain.get(current_state, [])

            if not possible_words:
                # If no transitions, pick a random start word
                current_state = random.choice(self.start_words)
                possible_words = self.chain.get(current_state, [])
                if not possible_words:
                    break

            # Choose the next word based on frequency for better accuracy
            next_word = max(possible_words, key=lambda word: self.word_counts[word])
            generated_words.append(next_word)
            current_state = tuple(generated_words[-self.n:])

        return ' '.join(generated_words)

app = Flask(__name__)

generator = PredictiveTextGenerator(n=3)
generator.case_sensitive = False  # Set to True to enable case sensitivity

# Load dataset from a file
def load_dataset(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    else:
        return ""

# Path to the dataset file
dataset_path = "dataset.txt"

# Train the model with the dataset or default text
training_text = """
Good morning! How are you today? I hope you have a great day ahead.
Don't forget to take a break and enjoy your lunch. Stay hydrated and keep smiling.
In the evening, relax and spend time with your loved ones. Have a peaceful night and sweet dreams.

Life is a journey, not a race. Take one step at a time and enjoy the little moments.
Remember to be kind to yourself and others. A smile can brighten someone's day.
Every day is a new opportunity to learn, grow, and make a difference.

Success is not the key to happiness. Happiness is the key to success. If you love what you are doing, you will be successful.
Challenges are what make life interesting, and overcoming them is what makes life meaningful.
Believe in yourself and all that you are. Know that there is something inside you that is greater than any obstacle.

The best way to predict the future is to create it. Dream big, work hard, and stay focused.
Mistakes are proof that you are trying. Learn from them and keep moving forward.
Every sunset brings the promise of a new dawn. Embrace each day with hope and positivity.
"""
dataset_content = load_dataset(dataset_path)
if dataset_content:
    generator.train(dataset_content.lower())
else:
    print("Dataset file not found. Using default training text.")
    generator.train(training_text.lower())

@app.route('/')
def index():
    return render_template('index.html', generator=generator)

@app.route('/predict', methods=['POST'])
def predict():
    user_input = request.form['text']
    seed_words = user_input.split()
    suggestions = []

    if len(seed_words) >= generator.n:  # Ensure the number of words matches the generator's n
        current_state = tuple(seed_words[-generator.n:])  # Use the last n words as the state
        possible_words = generator.chain.get(current_state, [])
        word_frequencies = {word: generator.word_counts[word] for word in possible_words}
        suggestions = sorted(word_frequencies, key=word_frequencies.get, reverse=True)[:3]  # Top 3 suggestions

    if not suggestions:
        suggestions = ["No suggestions available"]

    return {'suggestions': suggestions}

@app.route('/generate', methods=['POST'])
def generate():
    user_input = request.form['text']
    length = int(request.form.get('length', 50))  # Default length is 50 if not provided

    generated_text = generator.generate(length=length, seed_text=user_input)

    return {'generated_text': generated_text}

if __name__ == "__main__":
    app.run(debug=True)