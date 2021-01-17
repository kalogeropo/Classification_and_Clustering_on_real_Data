from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, adjusted_rand_score, fowlkes_mallows_score, v_measure_score
from sklearn.neighbors import NearestNeighbors

from meros_B import *
from meros_A import *
from partC import *

set_pandas_display_options()


filename_2 = '11.csv'
filename_1 = 'clinical_dataset.csv'
df1 = pd.read_csv(filename_2)
df2 = pd.read_csv(filename_1, sep=';')
df2 = cov_f(df2)  # todo  1
df2 = part_1(df2, j=0, high=0.95, low=0.005, thresh=3, x=2, na=35)  # todo  1

df_f = pd.merge(df1, df2, on='part_id')
df_f=df_f.drop(['part_id'], axis=1)
print(df_f.head(10))
sil_list=[0,0]
df_f2=df_f['fried']
for n_clusters in range(2,7):
    clusterer = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = clusterer.fit_predict(df_f)
    silhouette_avg = silhouette_score(df_f, cluster_labels)
    print("********************************************************")
    print("For n_clusters =", n_clusters,"The average silhouette_score is :", silhouette_avg)
    sil_list.append(silhouette_avg)
    cluster_analysis(cluster_labels,df_f2)

sc = StandardScaler()
x = sc.fit_transform(df_f)

plot_clusters_and_scores(sil_list,"silouete scores","No of Clusters","Silhouette Score on K clusters")

############################################################################
#Find optimal eps
#https://iopscience.iop.org/article/10.1088/1755-1315/31/1/012012/pdf
neigh = NearestNeighbors(n_neighbors=2)
nbrs = neigh.fit(df_f)
distances, indices = nbrs.kneighbors(df_f)
distances = np.sort(distances, axis=0)
distances = distances[:,1]
#plt.plot(distances)
#plt.show()
print(len(df_f2))
fmi_list = [0,0,0,0,0,0,0,0,0,0]
ari_list = [0,0,0,0,0,0,0,0,0,0]
v_m_list = [0,0,0,0,0,0,0,0,0,0]

for no_of_samples in range(10,40):
    clusterer = DBSCAN(eps=50,min_samples=no_of_samples,metric='euclidean',leaf_size=5)#DBSCAN(eps=0.5, min_samples=5, metric='euclidean', metric_params=None, algorithm='auto', leaf_size=30, p=None, n_jobs=None)

    cluster_labels = clusterer.fit_predict(df_f)
    ari = adjusted_rand_score(df_f2.values.tolist(), cluster_labels)
    #print(df_f2.values.tolist())
    #print(cluster_labels)
    fmi = fowlkes_mallows_score(df_f2.values.tolist(), cluster_labels)
    v_m = v_measure_score(df_f2.values.tolist(), cluster_labels)
    fmi_list.append(fmi)
    ari_list.append(ari)
    v_m_list.append(v_m)

    #print("ari =",ari)
    #print("Fmi=", fmi)
    #print("v_m=", v_m)

#plot  ari, fmi, V_m

#plot_clusters_and_scores(fmi_list,"Fowlkes Mallows scores","No of samples","Fowlkes Mallows scores on number of samples")
#plot_clusters_and_scores(ari_list,"Adjusted Rand Index scores","No of samples","Adjusted Rand Index on number of samples")
#plot_clusters_and_scores(v_m_list,"V measure","No of samples","V measure scores on number of samples")

