import pickle
from lm import LanguageModel

train_filename = "train_sequence.pkl"
model_filename = "model.pkl"

sequence = pickle.load(open(train_filename, "rb"))

lm = LanguageModel()
lm.fit(sequence)

pickle.dump(lm, open(model_filename, "wb"))
