import numpy as np

from .vocabulary import Vocabulary


class GloveEncoder(object):
    """Translates words to glove vectors."""

    def __init__(self, vocab_filename, vector_filename):
        """Initializes the GloveEncoder.

        Args:
            vocab_filename: Text file containing vocabulary.
            vector_filename: Text file containing GloVe embeddings.
        """
        self._vocab = Vocabulary()
        self._vocab.load(vocab_filename)
        self._embedding_matrix = self._load_embedding_matrix(vector_filename)
        print(self._embedding_matrix.shape)
        self._unk_embedding = np.mean(self._embedding_matrix, axis=0)
        print('Done loading glove')

    def _load_embedding_matrix(self, vector_filename):
        """Loads the embeddings matrix.

        Args:
            vector_filename: Text file containing GloVe embeddings.

        Returns:
            An np.array whose rows are GloVe vectors.
        """
        with open(vector_filename, 'r') as f:
            values = []
            for line in f:
                vector = line[:-1].split(' ')[1:]
                vector = [float(x) for x in vector]
                values.append(vector)
        embedding_matrix = np.array(values)
        return embedding_matrix

    def lookup(self, word):
        """Looks up the GloVe embedding for a given word.

        Args:
            word: Word to lookup GloVe vector for.

        Returns:
            An np.array containing the requested GloVe vector.
        """
        id = self._vocab.word_to_id(word)
        if id == self._vocab.unk_id:
            return self._unk_embedding
        else:
            return self._embedding_matrix[id,:]

