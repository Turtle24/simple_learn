from sklearn.datasets import load_iris
from simple_learn.classifiers import SimpleClassifier
from simple_learn.classifiers.pipeline import SimplePipeLine


iris = load_iris()
clf = SimpleClassifier()
pipe = SimplePipeLine(clf)

print(pipe.model_type)