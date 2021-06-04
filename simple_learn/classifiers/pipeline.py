from simple_learn.classifiers.simple_classifier import SimpleClassifier
from sklearn.pipeline import Pipeline
from pipe_grid import pipe_process_map

class SimplePipeLine:
    def __init__(self, model: SimpleClassifier(),) -> None:
        self.model_type = model.__class__.__name__
        self.preprocessor = tuple()
        self.score = 0

    def __repr__(self) -> str:
        pass

    def __str__(self) -> str:
        pass 

    def pipe_search(self):
        if self.model_type == 'SimpleClassifier':
            self.preprocessor = pipe_process_map['classifier'] 
        if self.model_type == 'SimpleRegressor':
            self.preprocessor = pipe_process_map['SimpleRegressor']
        else:
            raise f"The model type is {self.model_type}"
        
    def run_pipeline(self, model, X_train, y_train, X_test, y_test):
        best = {}
        prev = 0
        for pre in self.preprocessor:
            pipe = Pipeline([(pipe_process_map[pre])(str(model), model)])
            pipe.fit(X_train, y_train)
            curr = pipe.score(X_test, y_test)
            best[str(model)] = {'score': curr} if curr > prev else None 
            prev = curr
        
        

