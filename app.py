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

    def train(self, text):
        """
        Train the model on the input text
        """
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

generator = PredictiveTextGenerator(n=2)

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
Believe in the power of dreams and the strength of perseverance. Believe in the kindness of people and the beauty of life.
Believe in hard work, dedication, and the ability to overcome challenges. Believe in the potential of every new day to bring opportunities.

Be kind to everyone you meet, for you never know what battles they are fighting. Be kind to yourself and give yourself grace.
Kindness is a language that the deaf can hear and the blind can see. Be kind, for it costs nothing but means everything.

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
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    user_input = request.form['text']
    seed_words = user_input.split()

    # Ensure the user has typed at least three words
    if len(seed_words) >= 3:
        current_state = tuple(seed_words[-generator.n:])
        possible_words = generator.chain.get(current_state, [])
        suggestions = possible_words[:5]  # Return up to 5 suggestions
    else:
        suggestions = ["Please type at least three words to get predictions."]

    return {'suggestions': suggestions}

if __name__ == "__main__":
    app.run(debug=True)