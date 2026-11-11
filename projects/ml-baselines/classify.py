from sklearn.datasets import load_breast_cancer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X,y=load_breast_cancer(return_X_y=True,as_frame=True)
cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=17)
for name,model in [("majority",DummyClassifier(strategy="most_frequent")),("logistic",make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,class_weight="balanced")))]:
    result=cross_validate(model,X,y,cv=cv,scoring={"accuracy":"accuracy","precision":"precision","recall":"recall","f1":"f1","roc_auc":"roc_auc"})
    print(name,{k:round(result[f"test_{k}"].mean(),3) for k in ["accuracy","precision","recall","f1","roc_auc"]})
# A single holdout report makes errors concrete; it is not used for model selection.
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,stratify=y,random_state=17)
model=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,class_weight="balanced")).fit(Xtr,ytr)
print("Holdout ROC AUC:",round(roc_auc_score(yte,model.predict_proba(Xte)[:,1]),3))
print(classification_report(yte,model.predict(Xte),target_names=load_breast_cancer().target_names,zero_division=0))
