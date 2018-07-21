ALGORITHM_VERSION = 0.1

class Comparison:

    DATABASE = 'hash-database'
    db = []

    def __init__(self):
        """Connect to database"""
        from pymogo import Connection
        connection = Connection()
        self.db = connection[DATABASE]

    def get_similarity_score(self, new_hash, old_hashes):
        """Returns weighted average of cosine similarity"""
        date_created = [item["metadata"]["date_created"] for item in old_hashes]
        ages = [(time.now() - date) for date in date_created]
        weights = [(1.0 / age) for age in ages]

        def get_cosine_similarity(self, hash1, hash2):
            from sklearn.metrics.pairwise import cosine_similarity
            return float(cosine_similarity([hash1], [hash2]))

        score = [get_cosine_similarity(new_hash, old_hash) for old_hash in old_hashes]
        score = (scores * weights) / sum(weights)
        return score

    def find_student_hash(self, student_id):
        """Pulls hashes from database"""
        hashes = db.posts.find({"student_id": student_id})
        return hashes

    def insert_student_hash(self, new_hash):
        """Adds hash to database"""
        if new_hash.is_complete():
            db.posts.insert(new_hash)
        else:
            print("Hash incomplete, not added to database")






class Hash:
    'Uniqueness dictionary'
    """
    hash: {
        metadata: {
            student_id,
            genre,
            date_created,
            date_last_used,
            algorithm_version,
            num_comparisons,
            num_correct_comparisons
        }
        neural: [
            vector
        ]
        statistics: {
            word_count,
            rare_word_count,
            grammar_score
        }
    }

    """
    author_hash = {}

    def __init__(self):
        author_hash = {
            "metadata": {},
            "neural": [],
            "vector": {}
        }

    def is_complete(self):
        if author_hash["metadata"] &&
            author_hash["statistics"]  &&
            author_hash["neural"]:
            return True

    def create_hash(self, student_id, vector, genre=None, statistics=None):
        """Create hash with metadata, vector, and statistics"""

        author_hash["metadata"] = {
            "student_id": student_id,
            "genre": genre,
            "date_created": time.now(),
            "date_last_used": 0,
            "algorithm_version": ALGORITHM_VERSION,
            "num_comparisons": 0,
            "num_correct_comparisons": 0
        }
        author_hash["vector"] = vector
        if statistics:
            author_hash["statistics"] = statistics

    def set_hash(hash_from_db):
        author_hash = hash_from_db

class Origin:
    'Originality Algorithm'

    MAX_SEQUENCE_LENGTH = 500
    MODEL = '1-lstm-model.h5'

    def __init__(self):
        pass

    def get_hash(self):
        pass

    def compare_hash(self):
        pass

    def create_hash(self, text):
        """Preprocess text and return 100 character hash"""
        import keras
        from keras.models import Sequential

        def preprocess_text(self, text):
            """Tokenize and sequence text"""
            #from keras.preprocessing.text import Tokenizer
            from keras.preprocessing.sequence import pad_sequences
            import pickle

            with open('tokenizer.pickle', 'rb') as handle:
                tokenizer = pickle.load(handle)
            sequence = tokenizer.texts_to_sequences(text)
            sequence = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH)
            return sequence

        def get_prediction(self, sequence):
            """Load model and return prediction"""
            model = Sequential()
            model.load(MODEL)
            prediction = model.predict(sequence)
            return prediction[-1]

        sequence = preprocess_text(text)
        prediction = get_prediction(sequence)

        return
