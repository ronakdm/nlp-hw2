import pickle

START_TOKEN = "<START>"
UNK_TOKEN = "<UNK>"
STOP_TOKEN = "<STOP>"

train_in_filename = "data/1b_benchmark.train.tokens"
train_out_filename = "train_sequence.pkl"

dev_in_filename = "data/1b_benchmark.dev.tokens"
dev_out_filename = "dev_sequence.pkl"

test_in_filename = "data/1b_benchmark.test.tokens"
test_out_filename = "test_sequence.pkl"


def load_dataset(filename):

    with open(filename) as f:
        lines = f.readlines()

    # Pad the start and end of sentences with special tokens.
    sequence = []
    for line in lines:
        sequence.append([START_TOKEN, START_TOKEN] + line.split() + [STOP_TOKEN])

    return sequence


def format_dataset(in_filename, out_filename, freq):
    dataset = load_dataset(in_filename)
    for line in dataset:
        for i, token in enumerate(line):
            # Replace words not seen in the training set.
            if token not in freq or freq[token] < 3:
                line[i] = UNK_TOKEN

    pickle.dump(dataset, open(out_filename, "wb"))


# Training Set

dataset = load_dataset(train_in_filename)

# Count word frequency.
freq = {}
for line in dataset:
    for token in line:
        if token not in freq:
            freq[token] = 1
        else:
            freq[token] += 1

format_dataset(train_in_filename, train_out_filename, freq)

# Dev Set

format_dataset(dev_in_filename, dev_out_filename, freq)

# Test Set

format_dataset(test_in_filename, test_out_filename, freq)
