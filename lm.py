import numpy as np


class LanguageModel:
    def __init__(self, lambda1=0.1, lambda2=0.3, lambda3=0.6):
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3

        self.UNK_INDEX = 0
        self.STOP_INDEX = 1
        self.START_INDEX = 2

        self.UNK_TOKEN = "<UNK>"
        self.STOP_TOKEN = "<STOP>"
        self.START_TOKEN = "<START>"

        self.FREQ_TOKEN = "<FREQ>"
        self.INDEX_TOKEN = "<INDEX>"
        self.PROBS_TOKEN = "<PROBS>"

    def fit(self, sequence):
        self.word_tree, self.vocab_list = self._build_word_tree(sequence)

    def _build_word_tree(self, sequence):
        """
        Build a tree where branches are keyed by words.
        An n-length path corresponds to an n-gram.
        The <FREQ> special character will key the frequency of that n-gram.
        """
        word_tree = {
            self.UNK_TOKEN: {self.FREQ_TOKEN: 0, self.INDEX_TOKEN: self.UNK_INDEX},
            self.STOP_TOKEN: {self.FREQ_TOKEN: 0, self.INDEX_TOKEN: self.STOP_INDEX},
            self.START_TOKEN: {self.FREQ_TOKEN: 0, self.INDEX_TOKEN: self.START_INDEX},
        }
        vocab_list = [self.UNK_TOKEN, self.STOP_TOKEN, self.START_TOKEN]
        index = 3

        for t, token in enumerate(sequence):

            # Count unigram frequencies and assign indices to words.
            if token not in word_tree:
                word_tree[token] = {self.FREQ_TOKEN: 1, self.INDEX_TOKEN: index}
                index += 1
                vocab_list.append(token)
            else:
                word_tree[token][self.FREQ_TOKEN] += 1

            idx = word_tree[token][self.INDEX_TOKEN]

            # Count bigram frequencies.
            if t > 0:
                if token not in word_tree[sequence[t - 1]]:
                    word_tree[sequence[t - 1]][token] = {
                        self.FREQ_TOKEN: 1,
                        self.INDEX_TOKEN: idx,
                    }
                else:
                    word_tree[sequence[t - 1]][token][self.FREQ_TOKEN] += 1

            # Count trigram frequencies.
            if t > 1:
                if token not in word_tree[sequence[t - 2]][sequence[t - 1]]:
                    word_tree[sequence[t - 2]][sequence[t - 1]][token] = {
                        self.FREQ_TOKEN: 1,
                        self.INDEX_TOKEN: idx,
                    }
                else:
                    word_tree[sequence[t - 2]][sequence[t - 1]][token][
                        self.FREQ_TOKEN
                    ] += 1

        return word_tree, vocab_list

    def _build_probability_vector(self, node):
        """
        Take a node of the word tree (a dict) and compute the probability of the next
        word given the n-gram represented by that word. Putting in the root (self.word_tree) would 
        produce a unigram model probability vector, while self.word_tree[token1][token2] would
        produce p(...|token1, token2).
        """

        if self.PROBS_TOKEN in node:
            return node[self.PROBS_TOKEN]
        else:
            probs = np.zeros(len(self.vocab_list))
            for word in node:
                # Collect the frequencies of each of the words succeeding this one,
                # excluding special tokens.
                if word != self.FREQ_TOKEN and word != self.INDEX_TOKEN:
                    index = self.word_tree[word][self.INDEX_TOKEN]
                    probs[index] = node[word][self.FREQ_TOKEN]

            probs /= probs.sum()

            node[self.PROBS_TOKEN] = probs
            return probs

    def generate(self, seq_len):
        vocab_idx = np.arange(len(self.word_tree)).astype(int)
        sentence = []
        unigram_model = self._build_probability_vector(self.word_tree)

        for t in range(seq_len):

            # Generate probability of next work by computing the probability vector for each model.
            if t > 0:
                bigram_model = self._build_probability_vector(
                    self.word_tree[sentence[t - 1]]
                )
                if t > 1 and sentence[t - 1] in self.word_tree[sentence[t - 2]]:
                    trigram_model = self._build_probability_vector(
                        self.word_tree[sentence[t - 2]][sentence[t - 1]]
                    )
                    probs = (
                        self.lambda1 * unigram_model
                        + self.lambda2 * bigram_model
                        + self.lambda3 * trigram_model
                    ) / (self.lambda1 + self.lambda2 + self.lambda3)
                else:
                    probs = (
                        self.lambda1 * unigram_model + self.lambda2 * bigram_model
                    ) / (self.lambda1 + self.lambda2)
            else:
                probs = unigram_model

            token_idx = np.random.choice(vocab_idx, size=1, p=probs)[0]
            sentence.append(self.vocab_list[token_idx])

            # Check for start token <STOP> token.
            if token_idx == self.STOP_INDEX:
                break

        return " ".join(sentence)

    def compute_loglikelihood(self, sequence, base=np.exp(1)):

        unigram_model = self._build_probability_vector(self.word_tree)

        loglik = 0
        for t, token in enumerate(sequence):
            # Generate probability of next work by computing the probability vector for each model.
            if t > 0:
                bigram_model = self._build_probability_vector(
                    self.word_tree[sequence[t - 1]]
                )
                if t > 1 and sequence[t - 1] in self.word_tree[sequence[t - 2]]:
                    trigram_model = self._build_probability_vector(
                        self.word_tree[sequence[t - 2]][sequence[t - 1]]
                    )
                    probs = (
                        self.lambda1 * unigram_model
                        + self.lambda2 * bigram_model
                        + self.lambda3 * trigram_model
                    ) / (self.lambda1 + self.lambda2 + self.lambda3)
                else:
                    probs = (
                        self.lambda1 * unigram_model + self.lambda2 * bigram_model
                    ) / (self.lambda1 + self.lambda2)
            else:
                probs = unigram_model

            index = self.word_tree[token][self.INDEX_TOKEN]
            loglik += np.log(probs[index]) / np.log(base)

        return loglik

    def compute_perplexity(self, sequence, base=2):

        loglik = self.compute_loglikelihood(sequence, base=base)
        return base ** (-loglik / len(sequence))

