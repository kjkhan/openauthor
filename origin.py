import numpy as np
import time

DEBUG = True
ALGORITHM_VERSION = 0.1
MODEL = 'data/1-lstm32-model.h5'
TOKENIZER = 'data/tokenizer.pickle'
DATABASE = 'hash-database'
MAX_SEQUENCE_LENGTH = 500

def debug_print(text):
    if DEBUG:
        print(text)

class Interface:
    'Pass information to and from Canvas LMS'
    model = None

    def __init__(self):
        debug_print("Started interface")

    def score(self, id, text):
        new_hash = Origin().create_hash(id, text)
        score = Comparison().compare(new_hash)
        return score

class Database:
    db = None

    def __init__(self):
        """Connect to database"""
        debug_print("Connecting to database")
        self.connect()

    def connect(self):
        """Connect to Mongo database"""
        from pymongo import MongoClient
        from pymongo.errors import ConnectionFailure

        client = MongoClient()
        try:
            client.admin.command('ismaster')
        except ConnectionFailure:
            print("Database not available")

        self.db = client[DATABASE]

    def find(self, id):
        """Pulls hashes from database, convert to instances of Hash"""
        # TODO: update date_last_used
        finds = self.db.posts.find({"id": id})
        hashes = [Hash().set(h) for h in finds]
        return hashes

    def insert(self, new_hash):
        """Adds hash to database"""
        self.db.posts.insert(new_hash.get())

    def delete(self, id):
        self.db.posts.delete_many({"id": id})

class Comparison:
    'Compare hash with author previous hashes'
    def __init__(self):
        debug_print("Started comparison")

    def compare(self, new_hash):
        """Connect to database, find old hashes and insert new hash"""
        db = Database()
        old_hashes = db.find(new_hash.get_id())
        score, scores = self.score(new_hash, old_hashes)
        self.printout(new_hash, old_hashes, scores)
        db.insert(new_hash)
        return score

    def printout(self, new_hash, old_hashes, scores):
        from time import gmtime, strftime

        print("\nAuthor ID:", new_hash.get_id())
        print("Submitted text: '{}...'".format(new_hash.get_text()[0:51]))
        if scores == []:
            print("Score:  -1 (new author; nothing to compare)")
        else:
            print("Average score: {0:0.5f}".format(np.mean(scores)))
            print("\nHere's a comparision against prior submissions:")
            for i, old_hash in enumerate(old_hashes):
                date = strftime("%a %d %b %Y %H:%M:%S", gmtime(old_hash.get_date_created()))
                score = "{0:0.5f}".format(scores[i])
                print("{} ({}): {}: {}".format(i+1, date, old_hash.get_text()[0:51], score))

    def score(self, new_hash, old_hashes):
        """Return average similarity score or -1 if first"""
        if old_hashes == []:
            return -1, []

        def get_similarity(hash1, hash2):
            """Return sklearn's cosine_similarity for neural vectors"""
            from sklearn.metrics.pairwise import cosine_similarity
            neural1 = hash1.get_neural()
            neural2 = hash2.get_neural()
            score = float(cosine_similarity([neural1], [neural2]))
            return score

        scores = [get_similarity(new_hash, old_hash) for old_hash in old_hashes]
        score = np.mean(scores)
        return score, scores

        """TODO: add weighted averages
        date_created = [item["metadata"]["date_created"] for item in old_hashes]
        ages = [(time.now() - date) for date in date_created]
        weights = [(1.0 / age) for age in ages]
        score = (scores * weights) / sum(weights)
        """

class Hash:
    'Uniqueness dictionary'
    """
    hash: {
        id:
        metadata: {
            genre,
            date_created,
            date_last_used,
            algorithm_version,
            num_comparisons,
            num_correct_comparisons,
            text
        }
        neural: [
            vector
        ]
        statistics: {
            counts,
            readability
        }
    }

    """
    author_hash = {}

    def __init__(self):
        debug_print("Creating hash")
        self.author_hash = {
            "id": None,
            "metadata": None,
            "neural": None,
            "statistics": None
        }

    def is_complete(self):
        """Excludes statistics dictionary for now"""
        if self.author_hash["id"] is not None and \
            self.author_hash["metadata"] is not None and \
            self.author_hash["neural"] is not None:
            return True
        else:
            print("Hash is incomplete!")
            self.printout()

    def printout(self):
        print("Author ID:", self.author_hash["id"])
        print("Metadata:", self.author_hash["metadata"])
        print("Neural:", self.author_hash["neural"])
        print("Statistics:", self.author_hash["statistics"])

    def create(self, id, vector, statistics, text, genre=None):
        """Create hash with id, metadata, neural, and statistics"""
        self.author_hash["id"] = id
        self.author_hash["metadata"] = {
            "genre": genre,
            "date_created": time.time(),
            "date_last_used": 0,
            "algorithm_version": ALGORITHM_VERSION,
            "num_comparisons": 0,
            "num_correct_comparisons": 0,
            "text": text    # for beta use, to be removed on v1.0
        }
        self.author_hash["neural"] = vector.tolist()
        self.author_hash["statistics"] = statistics
        return self

    def set(self, hash_from_db):
        self.author_hash = hash_from_db
        return self

    def get(self):
        return self.author_hash

    def get_neural(self):
        """Update metadata and return neural vector"""
        self.author_hash["metadata"]["date_last_used"] = time.time()
        self.author_hash["metadata"]["num_comparisons"] += 1
        return self.author_hash["neural"]

    def get_id(self):
        return self.author_hash["id"]

    def get_text(self):
        return self.author_hash["metadata"]["text"]

    def get_date_created(self):
        return self.author_hash["metadata"]["date_created"]

class Origin:
    'Originality Algorithm'

    def __init__(self):
        debug_print("Executing originality algorithm")

    def create_neural(self, text):
        """Preprocess text and return neural output vector"""

        def preprocess_text(text):
            """Tokenize and sequence text"""
            #from keras.preprocessing.text import Tokenizer
            from tensorflow.keras.preprocessing.sequence import pad_sequences
            import pickle

            with open(TOKENIZER, 'rb') as handle:
                tokenizer = pickle.load(handle)
            sequence = tokenizer.texts_to_sequences(text)
            sequence = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH)
            return sequence

        def get_prediction(sequence):
            """Load model and return prediction"""
            from tensorflow import keras

            model = keras.models.load_model(MODEL)
            prediction = model.predict(sequence)
            return prediction[-1]

        sequence = preprocess_text(text)
        prediction = get_prediction(sequence)
        return prediction

    def create_statistics(self, text):
        """Get count-stats (# of words, unique words, long words, etc.) and
            readability stats (Flesch-Kincaid, etc.)"""
        import textacy

        doc = textacy.Doc(text, lang="en")
        ts = textacy.TextStats(doc)
        statistics = {
            "counts": ts.basic_counts,
            "readability": ts.readability_stats
        }
        return statistics

    def create_hash(self, id, text):
        """Create hash with neural and statistics input"""
        vector = self.create_neural(text)
        statistics = self.create_statistics(text)
        new_hash = Hash().create(id, vector, statistics, text)
        return new_hash
