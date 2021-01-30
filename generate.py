import pickle

model_filename = "model.pkl"

lm = pickle.load(open(model_filename, "rb"))

print(lm.generate(1000))
