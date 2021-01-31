import pickle
from lm import LanguageModel

train_filename = "train_sequence.pkl"
model_filename = "model.pkl"

dataset = pickle.load(open(train_filename, "rb"))

lm = LanguageModel(lidstone_param=3e-4)
lm.fit(dataset)

pickle.dump(lm, open(model_filename, "wb"))
