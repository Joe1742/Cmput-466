import multiprocessing
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

result = []  #elements are tuple:(accuracy, classifier, partial hyperparameters) 
cancer = load_breast_cancer() #load  breast cancer data 

#SGDClassifier: Stochastic Gradient Decreasing classifier
param_grid =[
	{
		'alpha': [1e-5,1e-4,5e-4,1e-3,2.3e-3,5e-3,1e-2],
		'penalty':['l2', 'l1', 'elasticnet'],
		'l1_ratio': [0.01,0.05,0.1,0.15,0.25,0.35,0.5,0.75,0.8]
	}]
sgd = SGDClassifier(loss='perceptron',learning_rate='optimal')
gscv = GridSearchCV(estimator=sgd, param_grid=param_grid,
	                scoring='accuracy', cv=10, n_jobs=multiprocessing.cpu_count())
gscv.fit(cancer.data, cancer.target)
result.append((gscv.best_score_,gscv.best_estimator_,gscv.best_params_))

#Descion Tree classifier
param_grid = [
{
		'criterion':["gini", "entropy", "log_loss"],
		'splitter': ["best", "random"]
	
}]
dtc = DecisionTreeClassifier()
gscv = GridSearchCV(estimator=dtc, param_grid=param_grid,
	                scoring='accuracy', cv=10, n_jobs=multiprocessing.cpu_count())
gscv.fit(cancer.data, cancer.target)
result.append((gscv.best_score_,gscv.best_estimator_,gscv.best_params_))

#Support Vector Machine classifier
param_grid = [{
		'kernel':['linear', 'poly', 'rbf', 'sigmoid'],
		'C': [0.1,0.2,0.4,0.5,1.0,1.5,1.8,2.0,2.5,3.0]
	}]
svc = SVC()
gscv = GridSearchCV(estimator=svc, param_grid=param_grid,
	                scoring='accuracy', cv=10, n_jobs=multiprocessing.cpu_count())
gscv.fit(cancer.data, cancer.target)
result.append((gscv.best_score_,gscv.best_estimator_,gscv.best_params_))

#Sort result and print	
result = sorted(result, key=lambda t:t[0], reverse=True)
for r in result:
	print(r)
