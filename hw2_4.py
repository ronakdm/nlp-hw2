import pickle

dev_filename = "dev_sequence.pkl"
test_filename = "test_sequence.pkl"
model_filename = "model.pkl"

dev_sequence = pickle.load(open(dev_filename, "rb"))
lm = pickle.load(open(model_filename, "rb"))

print("Dev set size:", len(dev_sequence))

dev_perplexity = lm.compute_perplexity(dev_sequence)
print("Dev set perplexity:", dev_perplexity)

