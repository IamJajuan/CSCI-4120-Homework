  
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from yellowbrick.cluster import KElbowVisualizer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import numpy as np
from scipy.stats import mode

def make_predicts(c,t,n):
    '''
    Make the predications
    '''
    labels = np.zeros_like(c)

    for i in range(n):

        mask = (c == i)
        labels[mask] = mode(t[mask])[0]
    
    return labels




def main():

    X, y_true = make_blobs(n_samples=300, centers=4,
                       cluster_std=0.60, random_state=0)
    
    model = KMeans()
    visualizer = KElbowVisualizer(model, k=(1,20))
    visualizer.fit(X)
    value = visualizer.elbow_value_
    model.n_clusters = value
    clusters = model.fit_predict(X)
    y_predict = make_predicts(clusters,y_true,4)
    matrix = confusion_matrix(y_true,y_predict)

    
    score = accuracy_score(y_true,y_predict)

    disp = ConfusionMatrixDisplay(confusion_matrix=matrix)
    disp.plot()
    visualizer.show()
    plt.show()
    



if __name__ == "__main__":

    main()

    
        