from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from experiment import load,metrics
from sklearn.linear_model import LogisticRegression,Ridge
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
for key in ['diabetes','housing']:
    d=load(key);tr,te=d['train'],d['test'];rows=[]
    models=[('LogisticRegression',LogisticRegression(max_iter=500,class_weight='balanced',random_state=42)),('RandomForest',RandomForestClassifier(n_estimators=100,max_depth=12,class_weight='balanced',random_state=42,n_jobs=2))] if key=='diabetes' else [('Ridge',Ridge(alpha=1)),('RandomForest',RandomForestRegressor(n_estimators=100,max_depth=12,random_state=42,n_jobs=2))]
    for name,model in models:
        model.fit(d['x'][tr],d['y'][tr]);pred=np.log(np.maximum(model.predict_proba(d['x'][te]),1e-12)) if key=='diabetes' else model.predict(d['x'][te])[:,None]
        rows.append(dict(Model=name,**metrics(d,te,pred)))
    pd.DataFrame(rows).to_csv(d['out']/'classical_baselines.csv',index=False);print(key,rows)
