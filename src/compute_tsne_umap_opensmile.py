import os, pickle, json
import numpy as np
from sklearn.manifold import TSNE
from umap import UMAP 
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import pandas as pd
#from utils import zscore
import scipy.stats as stats

LABEL = 'Opensmile-phrase_fold1'
direc = r'data\features\Saarbruecken\Multiclass\both\both_phrase\test_Multiclass_phrase_both_fold1.pkl'
direc_csv = r'data\features\Saarbruecken\Multiclass\both\both_phrase'
excel_metadata = r'data\lst\Saarbruecken\Saarbruecken_metadata.xlsx'
list_path = r'data\lst\Saarbruecken\Multiclass\both\both_phrase\test_Multiclass_phrase_meta_data_fold1.json'
    
    
# Read pathologies
labels_data = pd.read_excel(excel_metadata)
dclases = {label['File ID']: label['Medical_clasif'] for _,label in labels_data.iterrows()}
lab_pathology = []

# Read features
feature=[]


# Read labels
test_files = [] 
y, yy = [], [] 
testlist = list_path 
with open(testlist, 'r') as f:
    data = json.load(f)
    for item in data['meta_data']:
        test_files.append(item['path'])
        if item['label']=='0':
            y.append(0)
            yy.append('control')
        else:
            y.append(1)
            yy.append('pathology')        
        
        val = dclases[item['speaker']]
        kk=pd.read_csv(direc_csv+"\\"+str(item['speaker'])+"-phrase_smile.csv").to_numpy()[0]       
        feature.append(kk[3:].astype('float64'))        
        lab_pathology.append(val)        
        for nan in kk[3:]:
            if np.isnan(nan):
                print("inside", nan)
f.close()

print(feature)
# Plot image
X = np.array(feature)
plt.figure(1)
plt.pcolormesh(np.transpose(X))
plt.xlabel('utterance')
plt.ylabel('dimension')
plt.title(LABEL)
plt.colorbar()
#plt.savefig('Opensmile-'+LABEL+'.png')

# TSNE
X_embedded = TSNE(n_components=2, learning_rate='auto', init='pca', n_jobs=4, verbose=4).fit_transform(X)


palette = sns.color_palette("bright", 2)
color=[palette[i] for i in y]
df = pd.DataFrame({'x': X_embedded[:,0], 'y': X_embedded[:,1], 'label': yy, 'pathology': lab_pathology})
fig = px.scatter(df, x="x", y="y", color="label", hover_data=["pathology"])
fig.show()

plt.figure(2)
plt.subplot(121)
plt.title('TSNE')
plt.scatter(X_embedded[:,0], 
            X_embedded[:,1],
            c=[palette[i] for i in y])
plt.xlabel('TSNE dim 1')   
plt.ylabel('TSNE dim 2') 
plt.grid(True)
#plt.legend((LABEL+'-Fold1'))


# UMAP
reducer = UMAP()
U_embedded = reducer.fit_transform(X)

df = pd.DataFrame({'x': U_embedded[:,0], 'y': U_embedded[:,1], 'label': yy, 'pathology': lab_pathology})
fig = px.scatter(df, x="x", y="y", color="label", hover_data=["pathology"])
fig.show()

plt.subplot(122)
plt.title('UMAP')
plt.scatter(U_embedded[:,0], 
            U_embedded[:,1],
            c=[palette[i] for i in y])
#plt.legend((LABEL+'-Fold1'))
plt.xlabel('UMAP dim 1')   
plt.ylabel('UMAP dim 2') 
plt.grid(True)
plt.savefig('opensmile-tsne-umap-'+LABEL+'.png')
plt.show()





