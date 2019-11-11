############################################################
#modeling package
#programmed by: Cheng, Chang
#10/24/2019
#version: beta 1
############################################################

# class mdoel_selection packages
import time
import pandas as pd
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
#from tqdm import tqdm 
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
import plotly.express as px
from tqdm import tqdm_notebook as tqdm
import plotly.graph_objects as go

#class modeling_tools packages
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import numpy as np
import matplotlib.pyplot as plt
print('please type HOW_TO_USE() see instruction')

def HOW_TO_USE():
    i=1
    while True:
        which_class = input('which class you will use?\n1.model_selection\n2.modeling_tools\n3.confusionMatrix\n\nplease type 1, 2 or 3')
        if str(which_class) == '1':
            print('####################################################')
            print('########WELCOME TO MODEL SELECTION SYSTEM###########') 
            print('####################################################')
            print('example = select_model(X_train, y_train,kfold number (default 10), random state (default 1))')
            print('execute example.model_compare() directly then you can get result for model performance')
            print('\nif any question or comment, please email to Cheng Chang (cchang@cn.imshealth.com)')
            print('your advice will be most welcome and appreciate')
            print('####################################################')
            break
        elif str(which_class) == '2':
            print('####################################################')
            print('#########WELCOME TO MODEL ADJUSTMENT SYSTEM#########') 
            print('####################################################')
            print('\n here is how to use modeling_tools \n')
            print('example = modeling_tool(X_train, X_test, y_train, y_test, selected_clf,tuned_parameters=[])')
            print('tuned_parameters is a list like this')
            print('tuned_parameters = [{"learning_rate": [0.01,0.05], \n    "algorithm":["SAMME.R","SAMME"],.......}]')
            print('please check parms before use this function \n')

            print('####################################################')
            print('###############3 functions are included#############\n')
            print('1.use example.check_parm() to check what parameters include in current model')
            print('  after check, we can use example.update_parm([{"parmA":[1,2,3],....}]) to set our test parameter value\n')
            print('2.use example.test_estimator_num(start number, end number) to check best "estimator" for model\n')
            print('3.use example.test_GridSearchCV() to get best parameters')
            print('\nif any question or comment, please email to Cheng Chang (cchang@cn.imshealth.com)')
            print('your advice will be most welcome and appreciate')
            print('####################################################')
            break
        elif str(which_class) == '3':
            print('####################################################')
            print('##################CONFUSIONMATRIX###################') 
            print('####################################################') 
            print('after you fit model, use sklearn to make validation like y_pred = clf.predict(X_test).tolist()')  
            print('then call method confusionMatrix(y_pred, y_test).show()\n')   
            print("this method is copy from colleague's code")    
            print('Any question or comment, please email to Cheng Chang (cchang@cn.imshealth.com)')
            print('your advice will be most welcome and appreciate')
            print('####################################################') 
            break
        else:
            print('wrong value, please type again\n')
            i+=1
            if i==4:
                print('nope!!!! bye bye~')
                print('明·吴承恩《西游记》第二十七回：“常言道：‘事不过三。’我若不去，真是个下流无耻之徒。”')
                break


class select_model:

    def __init__(self, X_train, y_train,fold=10, seed=1):
        self.X_train = X_train
        self.y_train = y_train
        self.fold = fold
        self.seed = seed

    def model_compare(self):
        s = time.time()
        models = []
        models.append(('LR', LogisticRegression()))
        models.append(('LDA', LinearDiscriminantAnalysis()))
        models.append(('KNN', KNeighborsClassifier()))
        models.append(('CART', DecisionTreeClassifier()))
        models.append(('NB', GaussianNB()))
        models.append(('SVM', SVC()))
        models.append(('QDA', QuadraticDiscriminantAnalysis()))
        models.append(('RF', RandomForestClassifier()))
        models.append(('ADA', AdaBoostClassifier()))
        models.append(('GBC', GradientBoostingClassifier()))


        # evaluate each model in turn
        results = []
        names = []
        scoring = 'accuracy'
        print('fitting...')
        model_result_tab = pd.DataFrame()
        for name, model in tqdm(models):
            kfold = model_selection.KFold(n_splits=self.fold, random_state=self.seed)
            cv_results = model_selection.cross_val_score(model, self.X_train, self.y_train, cv=kfold, scoring=scoring)
            results.append(cv_results)
            names.append(name)
            model_result_tab[name] = pd.Series(cv_results)
            msg = "%s: %f (std: %f)" % (name, round(cv_results.mean(),2), round(cv_results.std(),2))
            print(msg)

        print('ploting...')
        fig = go.Figure()
        for col in model_result_tab.columns:
            fig.add_trace(go.Box(y=model_result_tab[col], name=col
                #,notched=True
                ))
        #set layout

        fig.update_layout(
            title=go.layout.Title(
                    text="MODEL PERFORMACE",
                    xref="paper"
                    #,size=25
                    #,x=0
                ),
                xaxis=go.layout.XAxis(
                    title=go.layout.xaxis.Title(
                        text="Model",
                        font=dict(
                        #    family="Courier New, monospace",
                            size=18,
                            color="#7f7f7f"
                        )
                    )
                ),
                yaxis=go.layout.YAxis(
                    title=go.layout.yaxis.Title(
                        text="Accuracy",
                        font=dict(
                        #    family="Courier New, monospace",
                            size=18,
                            color="#7f7f7f"
                        )
                    )
                ),
            #paper_bgcolor='rgb(104,104,104)',
            #plot_bgcolor='rgb(211,211,211)'
            )
        fig.show()
        e = time.time()
        print(f'finish in {round((e-s)/60,2)} minutes')

class modeling_tool:
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import classification_report
    import numpy as np
    import matplotlib.pyplot as plt
    def __init__(self,X_train, X_test, y_train, y_test, selected_clf, tuned_parameters=[]):
        self.tuned_parameters = tuned_parameters
        self.selected_clf = selected_clf
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
    
    def check_parm(self):
        print(self.selected_clf().get_params())

    def update_parm(self,new_list):
        self.tuned_parameters = new_list
        print(f'done! set test parameters to {new_list}')
        
    def test_GridSearchCV(self,scoring="accuracy",cv=10, verbose=30):
        #sklearn
        s = time.time()
        clf=GridSearchCV(self.selected_clf(),self.tuned_parameters,scoring=scoring,cv=cv, verbose=verbose)
        print('fitting...')
        clf.fit(self.X_train,self.y_train)
        print("Best parameters set found:",clf.best_params_)

        print("Optimized Score:",round(clf.score(self.X_test,self.y_test),3))
        print("Detailed classification report:")
        y_true, y_pred = self.y_test, clf.predict(self.X_test)
        print(classification_report(y_true, y_pred))
        e = time.time()
        print(f'finish in {round((e-s)/60,2)} minutes')

    def test_estimator_num(self,start,end):
        s = time.time()
        nums=np.arange(start,end,step=2)
        fig=plt.figure()
        ax=fig.add_subplot(1,1,1)
        testing_scores=[]
        training_scores=[]
        for num in tqdm(nums):
            clf=self.selected_clf(n_estimators=num)
            clf.fit(self.X_train,self.y_train)
            training_scores.append(clf.score(self.X_train,self.y_train))
            testing_scores.append(clf.score(self.X_test,self.y_test))
        ax.plot(nums,training_scores,label="Training Score")
        ax.plot(nums,testing_scores,label="Testing Score")
        ax.set_xlabel("estimator num")
        ax.set_ylabel("score")
        ax.legend(loc="lower right")
        ax.set_ylim(0,1.05)
        plt.suptitle(str(self.selected_clf))
        plt.show()
        e = time.time()
        print(f'finish in {round((e-s)/60,2)} minutes')


class confusionMatrix:
    """
    date: Mar 16, 2018
    This class aims to create confusion Matrix which has a similar style 
    in R (caret::confusionMatrix())    
    
    !!!! PLEASE NOTICE !!!!
    The first argument is the predicted values;
    The second argument is the actual values. 
    This is consistent with caret::confusionMatrix() in R,
    and opposite to confusion_matrix() in sklearn.
    
    .show() will print the results
    .result is a dictionary containing key information of the confusion matrix.
    """
    def __init__(self, predicted, actual):
        from sklearn.metrics import confusion_matrix
        import pandas as pd
        import numpy as np
        actual = pd.Series(actual)
        label = sorted(list(actual.value_counts().index))
        l = len(label)
        cfm = confusion_matrix(actual, predicted, labels = label)
        cfm = pd.DataFrame(cfm.T, columns = label, 
                           index = ['      ' + str(i) for i in label])
        precision = np.zeros(l)
        Recall = np.zeros(l)
        right = 0
        for i in range(l):
            precision[i] = (cfm.iloc[i,i] / sum(cfm.iloc[i,:]))
            Recall[i] = (cfm.iloc[i,i] / sum(cfm.iloc[:,i]))
            right = right + cfm.iloc[i,i]
            
        cfmStatistics = np.vstack((pd.DataFrame(precision).T,
                                   pd.DataFrame(Recall).T))
        cfmStatistics = pd.DataFrame(cfmStatistics, 
                                     columns = label,
                                     index = ['Precision','Recall'])
        Accuracy = round(right / sum(sum(np.array(cfm))), 4)
        NIR = round(max(actual.value_counts()) / sum(sum(np.array(cfm))), 4)
        self.result = {'cfm':cfm, 'Accuracy':Accuracy, 
                                  'NIR':NIR, 'Statistics':cfmStatistics}
        
    def show(self):
        print('Confusion Matrix and Statistics\n')
        print('           Reference')
        print('Prediction')
        print(self.result['cfm'], '\n')
        print('Overall Statistics\n')
        print('              Accuracy : ', self.result['Accuracy'])
        print('   No Information Rate : ', self.result['NIR'], '\n')        
        print('Statistics by Class:\n')
        print(self.result['Statistics'])
        

def nearZeroVar(raw2, freqCut = 99/1, uniqueCut = 10):
    """
    This function is similar with the one in R (caret::nearZeroVar) with fewer
    Arguments.
    
    
    Description
    
    nearZeroVar diagnoses predictors that have one unique value (i.e. are zero 
    variance predictors) or predictors that are have both of the following 
    characteristics: they have very few unique values relative to the number 
    of samples and the ratio of the frequency of the most common value to the 
    frequency of the second most common value is large.
    
    
    Arguments
    
    freqCut : the cutoff for the ratio of the most common value to the 
              second most common value.
    uniqueCut : the cutoff for the percentage of distinct values out of 
              the number of total samples
    """
    import numpy as np
    uniqueCut_1 = uniqueCut / 100 # be consistent with nearZeroVar in R(caret)
    n = raw2.columns
    todel = list()
    for col in n:
        if(raw2[col].dtypes == 'O'):
            if((list(raw2[col].value_counts())[0] / 
                list(raw2[col].value_counts())[1]) > freqCut):
                if(len(raw2[col].unique()) / len(raw2[col]) < uniqueCut_1):
                    todel.append(col)
                    continue
        else:
            if(np.var(raw2[col]) == 0):
                todel.append(col)
                continue
            if((list(raw2[col].value_counts())[0] / 
                list(raw2[col].value_counts())[1]) > freqCut):
                if(len(raw2[col].unique()) / len(raw2[col]) < uniqueCut_1):
                    todel.append(col)
                    continue
    return todel
      





