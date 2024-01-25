import pandas as pd
import math
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D


data = pd.read_csv('StressLevelDataset.csv')
print(data.shape)
print(data.dtypes)

####################################   analyse de données ###########################################

# Calculer le nombre de lignes et de colonnes nécessaires
num_cols = len(data.columns)
num_rows = math.ceil(num_cols / 4)

# Définir la taille de la figure
plt.figure(figsize=(15, 5 * num_rows))

# Parcourir chaque colonne du DataFrame
for i, col in enumerate(data.columns):
    # Créer un sous-plot pour chaque colonne
    plt.subplot(num_rows, 4, i + 1)

    # Créer un histogramme pour la colonne actuelle
    data[col].hist(bins=20, color='skyblue', edgecolor='black')

    # Ajouter des étiquettes et un titre
    plt.title(col)
    plt.xlabel("Valeurs")
    plt.ylabel("Fréquence")

# Ajuster l'espacement entre les sous-graphiques
plt.tight_layout()

# Afficher la figure
plt.show()

plt.figure(figsize=(12, 8))
data.boxplot()
plt.title("Diagrammes en boîte pour chaque colonne")
plt.xticks(rotation=45)
plt.show()


# Visualisez la distribution des données avec KDE pour chaque colonne
plt.figure(figsize=(15, 10))
sns.set(style="whitegrid")

# Parcourez toutes les colonnes de votre ensemble de données
for column in data.columns:
    sns.kdeplot(data[column], label=column, fill=True)

plt.title('Distribution des Caractéristiques avec KDE')
plt.xlabel('Valeurs')
plt.ylabel('Densité')
plt.legend()
plt.show()

data.drop_duplicates()
data.dropna()

#################################### Normalisation des données ###########################################


# Standardisation des données
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)
print(scaled_data[:5])

nom_variable_1 = 'anxiety_level'
nom_variable_2 = 'mental_health_history'
# Trouver l'indice des variables dans votre ensemble de données d'origine
indice_variable_original_1 = data.columns.get_loc(nom_variable_1)
indice_variable_original_2 = data.columns.get_loc(nom_variable_2)

# Visualisation avant et après la standardisation
plt.figure(figsize=(12, 6))

# Avant la standardisation
plt.subplot(1, 2, 1)
sns.histplot(data[nom_variable_1], kde=True, color='blue', label=nom_variable_1)
sns.histplot(data[nom_variable_2], kde=True, color='orange', label=nom_variable_2)
plt.title('Avant la Standardisation')
plt.xlabel('Valeurs')
plt.ylabel('Densité')
plt.legend()

# Après la standardisation
plt.subplot(1, 2, 2)
sns.histplot(scaled_data[:, indice_variable_original_1], kde=True, color='blue', label=nom_variable_1)
sns.histplot(scaled_data[:, indice_variable_original_2], kde=True, color='orange', label=nom_variable_2)
plt.title('Après la Standardisation')
plt.xlabel('Valeurs standardisées')
plt.ylabel('Densité')
plt.legend()

plt.tight_layout()
plt.show()


#################################### Application de K-means ###########################################


#copie du DataFrame
df_copie = data.copy()
from sklearn.metrics import silhouette_score

# Liste pour stocker les scores de silhouette
silhouette_scores = []

# On peut essayer différents nombres de clusters
for i in range(2, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    cluster_labels = kmeans.fit_predict(scaled_data)

    # Calcul du score de la silhouette
    silhouette_avg = silhouette_score(scaled_data, cluster_labels)
    silhouette_scores.append(silhouette_avg)

# Le nombre optimal de clusters avec le score de silhouette le plus élevé
optimal_clusters = silhouette_scores.index(max(silhouette_scores)) + 2  # +2 car le range commence à 2
print(f"Nombre optimal de clusters selon le score de silhouette : {optimal_clusters}")

# Graphique du score de silhouette
plt.plot(range(2, 11), silhouette_scores)
plt.title('Score de Silhouette')
plt.xlabel('Nombre de clusters')
plt.ylabel('Score de Silhouette moyen')
plt.show()

k = 5
kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=0)
df_copie['cluster'] = kmeans.fit_predict(scaled_data)

# Choisir le nombre de composantes pour le PCA(2D ou 3D)
n_components = 3

# Appliquer le PCA
pca = PCA(n_components=n_components)
pca_result = pca.fit_transform(scaled_data)

# Créer un DataFrame avec les composantes principales
pca_df = pd.DataFrame(data=pca_result, columns=[f'PC{i+1}' for i in range(n_components)])
pca_df['cluster'] = df_copie['cluster']

# Utiliser seaborn pour le plot avec des couleurs représentant les clusters
sns.scatterplot(x='PC1', y='PC2', hue='cluster', data=pca_df, palette='viridis', legend='full')
plt.title(f'PCA ')
plt.show()

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Appliquer le PCA avec 3 composantes
pca = PCA(n_components=3)
pca_result_3d = pca.fit_transform(scaled_data)

# Créer un DataFrame avec les trois composantes principales
pca_df_3d = pd.DataFrame(data=pca_result_3d, columns=[f'PC{i+1}' for i in range(3)])
pca_df_3d['cluster'] = df_copie['cluster']

# Créer une figure 3D
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

# Utiliser scatter pour le plot en 3D avec des couleurs représentant les clusters
scatter = ax.scatter(pca_df_3d['PC1'], pca_df_3d['PC2'], pca_df_3d['PC3'], c=pca_df_3d['cluster'], cmap='viridis')

ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
ax.set_title('PCA - Composantes Principales 3D')
ax.legend(*scatter.legend_elements(), title='Clusters')

plt.show()

#deuxième copie de données
df_copie2 = data.copy()

#Réduire la dimensionalité avec PCA
pca = PCA()
pca_result = pca.fit_transform(scaled_data)
#Choisissez le nombre optimal de composantes en fonction de la variance expliquée
# le nombre de composantes principales
n_components = 3

# Réduction de la dimensionnalité avec le nombre choisi de composantes principales
pca = PCA(n_components=n_components)
pca_result = pca.fit_transform(scaled_data)

#le Nb optimal de clusters: methode de la silhoutte
silhouette_scores = []
for i in range(2, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    cluster_labels = kmeans.fit_predict(pca_result)
    silhouette_avg = silhouette_score(pca_result, cluster_labels)
    silhouette_scores.append(silhouette_avg)

plt.plot(range(2, 11), silhouette_scores)
plt.title('Score de silhouette en fonction du nombre de clusters')
plt.xlabel('Nombre de clusters')
plt.ylabel('Score de silhouette moyen')
plt.show()

optimal_clusters = silhouette_scores.index(max(silhouette_scores)) + 2
print(f"Nombre optimal de clusters selon le score de silhouette : {optimal_clusters}")

kmeans_optimal = KMeans(n_clusters=optimal_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
data['cluster'] = kmeans_optimal.fit_predict(pca_result)

fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(pca_result[:, 0], pca_result[:, 1], pca_result[:, 2], c=data['cluster'], cmap='viridis')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
ax.set_title(f'K-Means Clustering (Nombre de clusters : {optimal_clusters})')
ax.legend(*scatter.legend_elements(), title='Clusters')
plt.show()

kmeans_optimal = KMeans(n_clusters=optimal_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
data['cluster'] = kmeans_optimal.fit_predict(pca_result)
# Visualisation des résultats en 2D
plt.figure(figsize=(5, 5))
scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], c=data['cluster'], cmap='viridis')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title(f'K-Means Clustering (Nombre de clusters : {optimal_clusters})')
plt.legend(*scatter.legend_elements(), title='Clusters')
plt.show()


#Analyse des résultats
# Ajoutez la colonne des clusters à votre DataFrame
df_copie2['cluster'] = kmeans_optimal.fit_predict(pca_result)

# Utilisez value_counts() pour obtenir le nombre d'individus dans chaque cluster
cluster_counts = df_copie2['cluster'].value_counts()


print(cluster_counts)
k = 4
kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=0)
df_copie2['cluster'] = kmeans.fit_predict(scaled_data)
cluster_summary = df_copie2.groupby('cluster').mean()
print(cluster_summary)


#################################### Application de DBSCAN ###########################################


#trouver la valeur d'epsilon
# Utiliser l'algorithme des k plus proches voisins pour calculer les distances
k = 5  # Choisis un nombre de voisins
nbrs = NearestNeighbors(n_neighbors=k).fit(scaled_data)
distances, _ = nbrs.kneighbors(scaled_data)

# Trier les distances des k plus proches voisins
distances = np.sort(distances, axis=0)
distances = distances[:,1]  # Prendre la distance au k plus proche voisin (exclut le point lui-même)

# Tracer le graphique des k-distances
plt.figure(figsize=(10, 6))
plt.plot(distances)
plt.title('Graphique des k-distances')
plt.xlabel('Points triés')
plt.ylabel(f'{k}-Distance')

# Ajouter des lignes verticales pour marquer plusieurs distances potentielles
thresholds = [ 0.65 , 1.7 , 3.75]  # Liste de distances à marquer
colors = ['r', 'g', 'b']  # Couleurs pour les lignes verticales

for threshold, color in zip(thresholds, colors):
    plt.axhline(y=threshold, color=color, linestyle='--', label=f'eps = {threshold}')

plt.legend()
plt.show()

#appliquer le PCA avant le clustering pour réduire la dimension
#Calcul de clusters
dbscan = DBSCAN(eps=0.65, min_samples=5)
clusters = dbscan.fit_predict(scaled_data)
# Exemple avec ACP
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(scaled_data)
clusters = dbscan.fit_predict(reduced_data)

# Nombre de clusters et points considérés comme du bruit (-1)
num_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
num_noise = list(clusters).count(-1)

print("Nombre de clusters :", num_clusters)
print("Nombre de points de bruit :", num_noise)


plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=clusters, cmap='viridis', marker='o', edgecolors='black')
plt.title('DBSCAN Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.colorbar(label='Cluster')
plt.show()

#coefficient de silhouette pour calculer la séparation
# Calculer la silhouette moyenne pour l'ensemble des données
average_silhouette = silhouette_score(reduced_data, clusters)

print("Silhouette moyenne :", average_silhouette)



# Ajouter le calcul du sommaire de cluster
data['cluster'] = clusters
cluster_summary = data.groupby('cluster').mean()
print(cluster_summary)

####### test 1 : différente valeur pour epsilon #############

#Appliquer le PCA après le clustering
# Appliquer DBSCAN sur les données normalisées
dbscan = DBSCAN(eps=1.7, min_samples=5)
clusters = dbscan.fit_predict(scaled_data)

# Nombre de clusters et points considérés comme du bruit (-1)
num_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
num_noise = list(clusters).count(-1)

print("Nombre de clusters :", num_clusters)
print("Nombre de points de bruit :", num_noise)

# Appliquer l'ACP sur les données normalisées
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(scaled_data)

# Visualisation des clusters en utilisant les composantes principales
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=clusters, cmap='viridis', marker='o', edgecolors='black')
plt.title('DBSCAN Clustering with PCA')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
# Ajouter la légende
plt.legend(*scatter.legend_elements(), title='Clusters')

plt.colorbar(label='Cluster')
plt.show()


##### test 2 : Application de PCA après le clustering #####
# Appliquer DBSCAN sur les données normalisées
dbscan = DBSCAN(eps=1.7, min_samples=5)
clusters = dbscan.fit_predict(scaled_data)

# Nombre de clusters et points considérés comme du bruit (-1)
num_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
num_noise = list(clusters).count(-1)

print("Nombre de clusters :", num_clusters)
print("Nombre de points de bruit :", num_noise)

# Appliquer l'ACP sur les données normalisées
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(scaled_data)

# Visualisation des clusters en utilisant les composantes principales
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=clusters, cmap='viridis', marker='o', edgecolors='black')
plt.title('DBSCAN Clustering with PCA')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
# Ajouter la légende
plt.legend(*scatter.legend_elements(), title='Clusters')

plt.colorbar(label='Cluster')
plt.show()


### calcul du coefficient dans ce cas

# Appliquer DBSCAN sur les données normalisées
dbscan = DBSCAN(eps=1.7, min_samples=5)
clusters = dbscan.fit_predict(scaled_data)

# Nombre de clusters et points considérés comme du bruit (-1)
num_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
num_noise = list(clusters).count(-1)

print("Nombre de clusters :", num_clusters)
print("Nombre de points de bruit :", num_noise)

# Calcul du coefficient de silhouette
silhouette_avg = silhouette_score(scaled_data, clusters)
print("Coefficient de silhouette moyen :", silhouette_avg)

# Ajouter le calcul du sommaire de cluster
data['cluster'] = clusters
cluster_summary = data.groupby('cluster').mean()
print(cluster_summary)



## La meilleur valeur pour DBSCAN
### Les paramètres de DBSCAN les plus convenables dans notre cas est  ε = 0.65 et MinPts = 5
### Le coefficient de silhouette dans ce cas est : 0.5890305702051282
### l'interprétation des clusters :
   #### cluster 0 : self_esteem elevée, anxiety et depression moyenne
   #### cluster 1: Depression et anxiety elevées, self_esteem moyen
   #### cluster 2: self_esteem elevé, anxiety moyenne et depression moyenne
