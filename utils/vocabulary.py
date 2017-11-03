"""Defines the vocabulary object which maps words in vocab files to integer IDs"""

import collections


class Vocabulary(object):

    def build_from_words(self, words, min_word_count=1):
        """Constructs vocabulary from a collection of words.

        Args:
            words: An iterable containing a collection of unique words.
            min_word_count: Minimum number of times a word must occur to be
                added to vocabulary.
        """
        # Count words.
        counter = collections.Counter()
        counter.update(words)

        # Filter and sort.
        word_counts = [x for x in counter.items() if x[1] >= min_word_count]
        word_counts.sort(key=lambda x: x[1], reverse=True)
        self._word_counts = word_counts

        # Create the vocab dictionary and reverse_vocab.
        self.unk_id =  len(word_counts)
        self._reverse_vocab = [x[0] for x in word_counts]
        self._vocab = {x: y for y, x in enumerate(self._reverse_vocab)}

    def build_from_word_counts(self, word_counts):
        # Create the vocab dictionary and reverse_vocab.
        self._word_counts = word_counts
        self.unk_id =  len(word_counts)
        self._reverse_vocab = [x[0] for x in word_counts]
        self._vocab = {x: y for y, x in enumerate(self._reverse_vocab)}

    def save(self, filename):
        """Saves the vocabulary to a text file.

        Args:
            filename: Text file to write vocabulary to.
        """
        with open(filename, 'w') as f:
            for word, count in self._word_counts:
                f.write('%s %i\n' % (word, count))

    def load(self, filename):
        """Loads vocabulary from a text file.

        Args:
            filename: Text file containing the vocabulary.
        """
        with open(filename, 'r') as f:
            # Read word counts.
            word_counts = [line.split() for line in f]
            word_counts = [(' '.join(l[:-1]), int(l[-1])) for l in word_counts]
            self._word_counts = word_counts

            # Create the vocab dictionary and reverse_vocab.
            self.unk_id =  len(word_counts)
            self._reverse_vocab = [x[0] for x in word_counts]
            self._vocab = {x: y for y, x in enumerate(self._reverse_vocab)}

    def word_to_id(self, word):
        """Gets the id for a given word. If word is not in the vocabulary
        returns the id for the <UNK> word instead.

        Args:
            word: Token to look up.
        """
        if word in self._vocab:
            return self._vocab[word]
        else:
            return self.unk_id

    def id_to_word(self, id):
        """Gets the word corresponding to a given id.

        Args:
            id: Id of word.
        """
        if id == self.unk_id:
            return '<UNK>'
        else:
            return self._reverse_vocab[id]

