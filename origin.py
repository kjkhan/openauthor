class AuthorHash:
    'Uniqueness string'

    LENGTH = 200
    NUM_DECIMALS = 4

    author_hash = {}

    def __init__(self):
        pass

    def create_hash(self, vector):
        """Convert vector into hash dictionary"""
        hash = {}
        author_hash["metadata"] = []
        author_hash["statistics"] = []
        author_hash["neural"] = vector
        return author_hash

    def retrieve_hash(self, hashed):


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
