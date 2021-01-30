import numpy as np


class LanguageModel:
    def __init__(self, lambda1=0.1, lambda2=0.3, lambda3=0.6, lidstone_param=1):
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.lidstone_param = lidstone_param

        self.UNK_INDEX = 0
        self.STOP_INDEX = 1
        self.START_INDEX = 2

        self.UNK_TOKEN = "<UNK>"
        self.STOP_TOKEN = "<STOP>"
        self.START_TOKEN = "<START>"

        self.FREQ_TOKEN = "<FREQ>"
        self.INDEX_TOKEN = "<INDEX>"
        self.PROBS_TOKEN = "<PROBS>"

    def fit(self, dataset):
        self.word_tree, self.vocab_list, self.vocab_size = self._build_word_tree(
            dataset
        )

    def _add_to_node(self, token, node, index):
        if token not in node:
            node[token] = {self.FREQ_TOKEN: 1, self.INDEX_TOKEN: index}
        else:
            node[token][self.FREQ_TOKEN] += 1

    def _build_word_tree(self, dataset):
        """
        Build a tree where branches are keyed by words.
        An n-length path corresponds to an n-gram.
        The <FREQ> special character will key the frequency of that n-gram.
        """
        word_tree = {
            self.UNK_TOKEN: {self.FREQ_TOKEN: 0, self.INDEX_TOKEN: self.UNK_INDEX},
            self.STOP_TOKEN: {self.FREQ_TOKEN: 0, self.INDEX_TOKEN: self.STOP_INDEX},
            self.START_TOKEN: {
                self.FREQ_TOKEN: 0,
                self.INDEX_TOKEN: self.START_INDEX,
                self.START_TOKEN: {},
            },
        }
        vocab_list = [self.UNK_TOKEN, self.STOP_TOKEN, self.START_TOKEN]
        index = 3

        for line in dataset:
            for t in range(2, len(line)):
                token = line[t]

                # Count unigram frequencies and assign indices to words.
                if token not in word_tree:
                    word_tree[token] = {
                        self.FREQ_TOKEN: 1,
                        self.INDEX_TOKEN: index,
                    }
                    index += 1
                    vocab_list.append(token)
                else:
                    word_tree[token][self.FREQ_TOKEN] += 1

                idx = word_tree[token][self.INDEX_TOKEN]

                # Count bigram and trigram frequencies.
                self._add_to_node(token, word_tree[line[t - 1]], idx)
                self._add_to_node(token, word_tree[line[t - 2]][line[t - 1]], idx)

        # # Add a stop token in the beginning to trigger adding to p(.|START), p(.|START, START)
        # for t, token in enumerate(([self.STOP_TOKEN] + sequence)[1:]):

        #     # Count unigram frequencies and assign indices to words.
        #     if token not in self.word_tree:
        #         self.word_tree[token] = {self.FREQ_TOKEN: 1, self.INDEX_TOKEN: index}
        #         index += 1
        #         vocab_list.append(token)
        #     else:
        #         self.word_tree[token][self.FREQ_TOKEN] += 1

        #     idx = self.word_tree[token][self.INDEX_TOKEN]

        #     # Count bigram frequencies.
        #     if self.word_tree[sequence[t - 1]] == self.STOP_TOKEN:
        #         self._add_to_node(token, self.word_tree[self.START_TOKEN], idx)
        #         self._add_to_node(
        #             token, self.word_tree[self.START_TOKEN][self.START_TOKEN], idx
        #         )
        #     else:
        #         self._add_to_node(token, self.word_tree[sequence[t - 1]], idx)

        #     # Count trigram frequencies.
        #     if t > 1 and self.word_tree[sequence[t - 1]] != self.STOP_TOKEN:
        #         if self.word_tree[sequence[t - 2]] == self.STOP_TOKEN:
        #             self._add_to_node(
        #                 token, self.word_tree[self.START_TOKEN][sequence[t - 1]], idx
        #             )
        #         else:
        #             self._add_to_node(
        #                 token, self.word_tree[sequence[t - 2]][sequence[t - 1]], idx
        #             )

        vocab_size = len(word_tree)
        return word_tree, vocab_list, vocab_size

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
            probs = self.lidstone_param * np.ones(self.vocab_size)
            for word in node:
                # Collect the frequencies of each of the words succeeding this one,
                # excluding special tokens.
                if word not in [self.FREQ_TOKEN, self.INDEX_TOKEN, self.START_TOKEN]:
                    index = self.word_tree[word][self.INDEX_TOKEN]
                    probs[index] += node[word][self.FREQ_TOKEN]

            probs[self.START_INDEX] = 0
            probs /= probs.sum()

            node[self.PROBS_TOKEN] = probs
            return probs

    def generate(self, seq_len):
        vocab_idx = np.arange(self.vocab_size).astype(int)
        sentence = [self.START_TOKEN, self.START_TOKEN]
        unigram_model = self._build_probability_vector(self.word_tree)
        ignore_trigram = False
        for t in range(2, seq_len + 2):

            bigram_model = self._build_probability_vector(
                self.word_tree[sentence[t - 1]]
            )
            if sentence[t - 1] in self.word_tree[sentence[t - 2]]:
                trigram_model = self._build_probability_vector(
                    self.word_tree[sentence[t - 2]][sentence[t - 1]]
                )
            else:
                ignore_trigram = True
            # # Generate probability of next work by computing the probability vector for each model.
            # if t > 0:
            #     bigram_model = self._build_probability_vector(
            #         self.word_tree[sentence[t - 1]]
            #     )
            # else:
            #     bigram_model = self._build_probability_vector(
            #         self.word_tree[self.START_TOKEN]
            #     )
            # if t > 1:
            #     if sentence[t - 1] in self.word_tree[sentence[t - 2]]:
            #         trigram_model = self._build_probability_vector(
            #             self.word_tree[sentence[t - 2]][sentence[t - 1]]
            #         )
            #     else:
            #         trigram_model = np.ones(self.vocab_size) / self.vocab_size
            # else:
            #     trigram_model = self._build_probability_vector(
            #         self.word_tree[self.START_TOKEN][self.START_TOKEN]
            #     )

            if ignore_trigram:
                probs = (self.lambda1 * unigram_model + self.lambda2 * bigram_model) / (
                    self.lambda1 + self.lambda2
                )
            else:
                probs = (
                    self.lambda1 * unigram_model
                    + self.lambda2 * bigram_model
                    + self.lambda3 * trigram_model
                ) / (self.lambda1 + self.lambda2 + self.lambda3)

            token_idx = np.random.choice(vocab_idx, size=1, p=probs)[0]
            sentence.append(self.vocab_list[token_idx])

            # Check for <STOP> token.
            if token_idx == self.STOP_INDEX:
                break

        return " ".join(sentence)

    def compute_loglikelihood(self, dataset, base=np.exp(1)):

        unigram_model = self._build_probability_vector(self.word_tree)

        loglik = 0
        for line in dataset:
            for t in range(2, len(line)):
                bigram_model = self._build_probability_vector(
                    self.word_tree[line[t - 1]]
                )
                if line[t - 1] in self.word_tree[line[t - 2]]:
                    trigram_model = self._build_probability_vector(
                        self.word_tree[line[t - 2]][line[t - 1]]
                    )
                else:
                    trigram_model = np.ones(self.vocab_size) / self.vocab_size
                # for t, token in enumerate(sequence):
                #     # Generate probability of next work by computing the probability vector for each model.
                #     if t > 0:
                #         bigram_model = self._build_probability_vector(
                #             self.word_tree[sequence[t - 1]]
                #         )
                #     else:
                #         bigram_model = self._build_probability_vector(
                #             self.word_tree[self.START_TOKEN]
                #         )
                #     if t > 1:
                #         if sequence[t - 1] in self.word_tree[sequence[t - 2]]:
                #             trigram_model = self._build_probability_vector(
                #                 self.word_tree[sequence[t - 2]][sequence[t - 1]]
                #             )
                #         else:
                #             trigram_model = np.ones(self.vocab_size) / self.vocab_size
                #     else:
                #         trigram_model = self._build_probability_vector(
                #             self.word_tree[self.START_TOKEN][self.START_TOKEN]
                #         )

                probs = (
                    self.lambda1 * unigram_model
                    + self.lambda2 * bigram_model
                    + self.lambda3 * trigram_model
                ) / (self.lambda1 + self.lambda2 + self.lambda3)

                index = self.word_tree[line[t]][self.INDEX_TOKEN]
                loglik += np.log(probs[index]) / np.log(base)

        return loglik

    def compute_perplexity(self, dataset, base=2):

        num_words = 0
        for line in dataset:
            num_words += len(line) - 2
        loglik = self.compute_loglikelihood(dataset, base=base)
        return base ** (-loglik / num_words)

