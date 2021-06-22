from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV, cross_val_score


def build_model(x,y):

  forest = RandomForestClassifier(random_state=40,max_depth=20,min_samples_leaf=10,n_estimators=80)
  forest.fit(x, y)

  return forest


def main():

    init_data = load_breast_cancer()
    (X, y) = load_breast_cancer(return_X_y=True)
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
    forest = build_model(X_train,Y_train)
    model = SelectFromModel(forest,prefit=True,max_features=2)
    x_train = model.transform(X_train)
    x_test = model.transform(X_test)
    forest = build_model(x_train,Y_train)
    cur_score  = cross_val_score(forest,x_test,Y_test,cv=5)
    score = cur_score.mean()
    accuracy_per_feature = score/forest.n_features_
    print(f"Score {score}\nN_features {forest.n_features_}\nScore Per Feature {accuracy_per_feature}\n")


if __name__ == "__main__":

    main()

