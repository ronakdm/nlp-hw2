import pickle

train_filename = "train_sequence.pkl"
dev_filename = "dev_sequence.pkl"
test_filename = "test_sequence.pkl"
model_filename = "model.pkl"

sequence = pickle.load(open(dev_filename, "rb"))
lm = pickle.load(open(model_filename, "rb"))

print("Size:", len(sequence))

perplexity = lm.compute_perplexity(sequence)
print("Perplexity:", perplexity)

