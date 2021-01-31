import pickle

model_filename = "model.pkl"

datasets = [
    "train",
    "dev",
    "test",
]

models = [
    "unigram",
    "bigram",
    "trigram",
]


lm = pickle.load(open(model_filename, "rb"))
kwargs = {
    "unigram": {"lambda1": 1, "lambda2": 0, "lambda3": 0},
    "bigram": {"lambda1": 0, "lambda2": 1, "lambda3": 0},
    "trigram": {"lambda1": 0, "lambda2": 0, "lambda3": 1},
}
results = pickle.load(open("nonsmooth_results", "rb"))
for dataset in datasets:
    for model in models:
        filename = "%s_sequence.pkl" % dataset
        sequence = pickle.load(open(filename, "rb"))
        perplexity = lm.compute_perplexity(sequence, **kwargs[model], verbose=True)

        print(
            "%s model + %s dataset = %0.03f perplexity." % (model, dataset, perplexity)
        )
        results[model][dataset + "_perplexity"] = perplexity

pickle.dump(results, open("nonsmooth_results", "wb"))

