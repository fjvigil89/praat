import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth, SpectralClustering, KMeans,  AffinityPropagation
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from itertools import cycle
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import json


def cluster(X, y, name_label, path):        
    
    X = np.array(X)
    y = np.array(y)    
    
    sc = KMeans(n_clusters=10, random_state=0).fit(X)
    
    
    X = StandardScaler().fit_transform(X)
    # sc= SpectralClustering(affinity='rbf', assign_labels='kmeans', coef0=1, degree=3,
    #                 eigen_solver=None, eigen_tol=0.0, gamma=1.0,
    #                 kernel_params=None, n_clusters=4, n_components=None,
    #                 n_init=10, n_jobs=None, n_neighbors=10, random_state=None).fit(X)
    #histograma de cada claster
    labels = sc.labels_
    
    count= np.zeros(len(labels), dtype=int)
    for i in range(0,len(labels)):
        count[labels[i]]+=1
    
    
    data={}       
    cluster=[]
    for i in range(0,len(labels)):        
        data[labels[i]]= {'count': count[labels[i]]}
        cluster.append({'cluster': labels[i], 'pathology': name_label[i], 'group': y[i], 'path': path[i] })
    
    print("data",data)        
    
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print("Estimated number of clusters: %d" % n_clusters_)
    print("Estimated number of noise points: %d" % n_noise_)
    # print("Homogeneity: %0.3f" % metrics.homogeneity_score(y, labels))
    # print("Completeness: %0.3f" % metrics.completeness_score(y, labels))
    # print("V-measure: %0.3f" % metrics.v_measure_score(y, labels))
    # print("Adjusted Rand Index: %0.3f" % metrics.adjusted_rand_score(y, labels))
    # print(
    #     "Adjusted Mutual Information: %0.3f"
    #     % metrics.adjusted_mutual_info_score(y, labels)
    # )
    # print("Silhouette Coefficient: %0.4f" % metrics.silhouette_score(X, labels))
    
    fig = plt.figure(figsize=(int(n_clusters_)+6, int(n_clusters_)+6))
    
    for i in range(0,int(n_clusters_)):
        labels_unique = np.unique(name_label)
        count= np.zeros(len(labels_unique), dtype=int)
        k=0
        for item in cluster:
            if item['cluster'] == i:                                
                # count[np.where(labels_unique == item['group'])[0]] +=1        
                count[np.where(labels_unique == item['pathology'])[0]] +=1        
                k+=1
        plot_count=[]
        for j in count:            
            plot_count.append(j/k)
        
        
        ax= plt.subplot(3,5,1+i)
        
        plt.bar(labels_unique, plot_count)                
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        #plt.grid(True)

        plt.tight_layout()
        # plt.suptitle('Datos')
    
    plt.show()
    
    # for i, rect in enumerate(g):
    #     posx = rect.get_x()
    #     posy = rect.get_height()
    #     ax.text(posx + 0.03, posy + 30, int(v[i]), color='black', fontsize = 8)
    # plt.show()
    
    # fig, ax = plt.subplots()    
    # plt.scatter(X[:,0], y, c=labels)
    # plt.show()
