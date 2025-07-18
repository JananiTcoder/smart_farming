import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, log_loss,
    roc_auc_score, average_precision_score, brier_score_loss,
    RocCurveDisplay, PrecisionRecallDisplay, ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt

def evaluate_model(model,x_train, x_test, y_train, y_test, average='macro'):
    print("Performance Metrics:")
    y_pred=model.predict(x_test)
    y_proba=model.predict_proba(x_test)
    print("Accuracy:",accuracy_score(y_test,y_pred))
    print("Precision",precision_score(y_test,y_pred,average=average))
    print("Recall",recall_score(y_test,y_pred,average=average))
    print("F1 Score",f1_score(y_test,y_pred,average=average))
    print("Classification Report:\n",classification_report(y_test, y_pred))
    print("confusion_matrix:\n",confusion_matrix(y_test,y_pred))
    print("Log Loss:",log_loss(y_test,y_proba))
    print("ROC AUC Score(OVR):",roc_auc_score(y_test,y_proba,multi_class='ovr',average=average))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.title("Confusion Matrix")
    plt.grid(False)
    plt.show()
    classes=sorted(list(set(y_test)))
    y_test_bin=label_binarize(y_test,classes=classes)
    y_score=model.predict_proba(x_test)
    fpr=dict()
    tpr=dict()
    roc_auc=dict()
    for i in range(len(classes)):
        fpr[i],tpr[i], _ =roc_curve(y_test_bin[:,i],y_score[:,i])
        roc_auc[i]=auc(fpr[i],tpr[i])
    plt.figure()
    colors=['aqua','darkorange','cornflowerblue']
    for i,color in zip(range(len(classes)),colors):
        plt.plot(fpr[i],tpr[i],color=color,
                 label=f"Class{i}(AUC={roc_auc[i]:.2f})")
    plt.plot([0,1],[0,1],'k--',lw=1)
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Multiclass ROC Curve (One-vs-Rest)")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()
    classes=sorted(list(set(y_test)))
    y_test_bin=label_binarize(y_test,classes=classes)
    y_score=model.predict_proba(x_test)
    precision=dict()
    recall=dict()
    avg_precision=dict()
    for i in range(len(classes)):
        precision[i],recall[i],_=precision_recall_curve(y_test_bin[:,i],y_score[:,i])
        avg_precision[i]=average_precision_score(y_test_bin[:,i],y_score[:,i])
    plt.figure()
    colors=['navy','turquoise','darkorange']
    for i,color in zip(range(len(classes)),colors):
        plt.plot(recall[i],precision[i],color=color,
                label=f"Class{i}(AP={avg_precision[i]:.2f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Multiclass Precision-Recall Curve (OvR)")
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.show()


df=pd.read_csv("best_crop.csv")
le=LabelEncoder()
df["crop_type"]=le.fit_transform(df["crop_type"])
x=df.drop(["crop_type"],axis=1)
y=df["crop_type"]
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=0)
model=xgb.XGBClassifier(n_estimators=1000,max_depth=2,learning_rate=0.05,subsample=0.8,colsample_bytree=0.8,reg_alpha=5,reg_lambda=10,eval_metric='merror',early_stopping_rounds=10,random_state=42)
model.fit(xtrain,ytrain,eval_set=[(xtest,ytest)],verbose=True)
evaluate_model(model,xtrain,xtest,ytrain,ytest)