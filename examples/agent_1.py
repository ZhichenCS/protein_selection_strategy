import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Needed for projection='3d'
from sklearn.manifold import TSNE
from sklearn import preprocessing

import os
from sklearn.svm import OneClassSVM

import yaml
os.chdir(os.path.dirname(__file__))

np.random.seed(2934)
## todo: normalization before learning

def plot_comparison(components, X, y, dim_pref=2, t_sne=False):
    """Draw a scatter plot of points, colored by their labels, before and after applying a learned transformation

    Parameters
    ----------
    components : array_like
        The learned transformation in an array with shape (n_components, n_features).
    X : array_like
        An array of data samples with shape (n_samples, n_features).
    y : array_like
        An array of data labels with shape (n_samples,).
    dim_pref : int
        The preferred number of dimensions to plot (default: 2).
    t_sne : bool
        Whether to use t-SNE to produce the plot or just use the first two dimensions
        of the inputs (default: False).

    """
    if dim_pref < 2 or dim_pref > 3:
        print('Preferred plot dimensionality must be 2 or 3, setting to 2!')
        dim_pref = 2

    if t_sne:
        print("Computing t-SNE embedding")
        tsne = TSNE(n_components=dim_pref, init='pca', random_state=0)
        X_tsne = tsne.fit_transform(X)
        Lx_tsne = tsne.fit_transform(X.dot(components.T))
        X = X_tsne
        Lx = Lx_tsne
    else:
        Lx = X.dot(components.T)

    fig = plt.figure()
    if X.shape[1] > 2 and dim_pref == 3:
        ax = fig.add_subplot(121, projection='3d')
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y)
        ax.set_title('Original Data')
        ax = fig.add_subplot(122, projection='3d')
        ax.scatter(Lx[:, 0], Lx[:, 1], Lx[:, 2], c=y)
        ax.set_title('Transformed Data')
    elif X.shape[1] == 2:
        ax = fig.add_subplot(121)
        ax.scatter(X[:, 0], X[:, 1], c=y)
        ax.set_title('Original Data')
        ax = fig.add_subplot(122)
        ax.scatter(Lx[:, 0], Lx[:, 1], c=y)
        ax.set_title('Transformed Data')
        
    plt.show()
    plt.savefig(cfg[dataset_name]['fig_save_path']+'3dvisualCompare.png')


def read_rest_data(fpath):
    
    df = pd.read_csv(fpath)
    
    # names_feature = ['Pos 1', 'Pos 2', 'Pos 3', 'score', 'target', 'A1', 'B1', 'C1', 'A2', 'B2', 'C2', 'A3', 'B3', 'C3',
    # 'Final_intermolecular_energy', 'vdW_Hbond_desolv', 'Electrostatic_Energy', 'Final_Total_Internal_Energy', 'kd']
    used_feature = ['A', 'B', 'C']
    
    x = np.array(df[used_feature])
    names = np.array(df[['resn']])
    
    # standard_scaler = preprocessing.StandardScaler()
    # x = standard_scaler.fit_transform(x)
    
    return x, names
    
def read_dataset(fpath, shuffle=False, augment=False):
    df = pd.read_csv(fpath)
    
    # names_feature = ['Pos 1', 'Pos 2', 'Pos 3', 'score', 'target', 'A1', 'B1', 'C1', 'A2', 'B2', 'C2', 'A3', 'B3', 'C3',
    # 'Final_intermolecular_energy', 'vdW_Hbond_desolv', 'Electrostatic_Energy', 'Final_Total_Internal_Energy', 'kd']
    used_feature = ['Pos 1', 'Pos 2', 'Pos 3']
                    
    
    xy = df.loc[df['target'] != 'unknown'][:]
    x = np.array(xy[used_feature])
    y = np.array([1 if v=='good' else 0 for v in xy['target']])
    
    if shuffle:
        perm_idx = np.arange(len(y))
        np.random.shuffle(perm_idx)
        train_ind = perm_idx[0: int(len(y)*0.9)]
        val_ind = perm_idx[int(len(y)*0.9):]
    
        train_X, train_Y = x[train_ind], y[train_ind]
        val_X, val_Y = x[val_ind], y[val_ind]
    
    else:
        train_X, train_Y = x,y
        val_X, val_Y = [], []
        
    
    if augment:
        x_0 = train_X[train_Y==0]
        
        N = 400
        new_data = list()
        for i in range(N):
            ind_a = np.random.choice(len(x_0))
            ind_b = np.random.choice(len(x_0))
            w = np.random.uniform(0,1)
            c = w*x_0[ind_a] + (1-w)*x_0[ind_b]
            new_data.append(c)
        
        train_X = np.concatenate((train_X, new_data))
        
        new_y = np.array([0 for i in range(N)])
        train_Y = np.concatenate((train_Y, new_y))
        
    
    
    test_x = np.array(df.loc[df['target'] == 'unknown'][used_feature])
    test_name = np.array(df.loc[df['target'] == 'unknown'][['resn']])
    
    
    
    return train_X, train_Y, val_X, val_Y, test_x, test_name

def load_dataset(dataset_name, cfg):
    if dataset_name == 'PTS1':
        train_X, train_Y, val_X,  val_Y, test_x, test_name, rest_data, rest_names, rest_Y = read_dataset_PTS1(cfg)
    elif dataset_name == 'GB1':
        train_X, train_Y, val_X,  val_Y, test_x, test_name, rest_data, rest_names, rest_Y = read_dataset_GB1(cfg)
    elif dataset_name == 'PhoQ':
        train_X, train_Y, val_X,  val_Y, test_x, test_name, rest_data, rest_names, rest_Y = read_dataset_PhoQ(cfg)
    elif dataset_name == 'GB1_score3_consider1ps':
        train_X, train_Y, val_X,  val_Y, test_x, test_name, rest_data, rest_names, rest_Y = read_dataset_GB1_score3_consider1ps(cfg)
    elif dataset_name == 'GB1_set1ps_change2ps_consider1ps':
        train_X, train_Y, val_X,  val_Y, test_x, test_name, rest_data, rest_names, rest_Y = read_dataset_GB1_set1ps_change2ps_consider1ps(cfg)
        
    
    used_features = []
    return train_X, train_Y, val_X,  val_Y, test_x, test_name, rest_data, rest_names, rest_Y
    
    
    
def read_dataset_PTS1(cfg):
    fpath = cfg['PTS1']['dataset_path']
    rest_fpath = cfg['PTS1']['testset_path']
    train_X, train_Y, val_X,  val_Y, test_x, test_name= read_dataset(fpath, shuffle=True, augment=True)
    rest_data, rest_names = read_rest_data(rest_fpath)
    rest_label = np.ones_like(rest_names)
    print('*'*20+'fake test_label!')
    
    return train_X, train_Y, val_X,  val_Y, test_x, test_name, rest_data, rest_names, rest_label


def read_dataset_GB1(cfg):
    # * input: read p1, p2, p3, p4, f1,f2,f3, f4
    # * output 
    
    n_f = 4
    GB1 = np.load(cfg['GB1']['processed_data_path']+'Pre_GB1.npz', allow_pickle=True)
    train_X, train_Y = GB1['train_X'], GB1['train_Y']
    val_X, val_Y = GB1['val_X'], GB1['val_Y']
    if len(val_X) == 0:
        val_X, val_Y = train_X, train_Y
    
    test_X, test_Y = GB1['test_X'], GB1['test_Y']
    test_names = GB1['test_names']
           
    return train_X[:, 0:n_f], train_Y, val_X[:, 0:n_f], val_Y, val_X[:, 0:n_f], val_Y, test_X[:, 0:n_f], test_names, test_Y,

def read_dataset_PhoQ(cfg):
    n_f = 4
    GB1 = np.load(cfg['PhoQ']['processed_data_path']+'Pre_PhoQ.npz', allow_pickle=True)
    train_X, train_Y = GB1['train_X'], GB1['train_Y']
    val_X, val_Y = GB1['val_X'], GB1['val_Y']
    if len(val_X) == 0:
        val_X, val_Y = train_X, train_Y
    
    test_X, test_Y = GB1['test_X'], GB1['test_Y']
    test_names = GB1['test_names']
           
    return train_X[:, 0:n_f], train_Y, val_X[:, 0:n_f], val_Y, val_X[:, 0:n_f], val_Y, test_X[:, 0:n_f], test_names, test_Y,
    

def read_dataset_GB1_score3_consider1ps(cfg):
    n_f = 4
   
    GB1 = np.load(cfg['GB1_score3_consider1ps']['processed_data_path']+'Pre_PhoQ.npz', allow_pickle=True)
    train_X, train_Y = GB1['train_X'], GB1['train_Y']
    val_X, val_Y = GB1['val_X'], GB1['val_Y']
    if len(val_X) == 0:
        val_X, val_Y = train_X, train_Y
    
    test_X, test_Y = GB1['test_X'], GB1['test_Y']
    test_names = GB1['test_names']
           
    return train_X[:, 0:n_f], train_Y, val_X[:, 0:n_f], val_Y, val_X[:, 0:n_f], val_Y, test_X[:, 0:n_f], test_names, test_Y,

def read_dataset_GB1_set1ps_change2ps_consider1ps(cfg):
    n_f = 4
   
    GB1 = np.load(cfg['GB1_set1ps_change2ps_consider1ps']['processed_data_path']+'Pre_PhoQ.npz', allow_pickle=True)
    train_X, train_Y = GB1['train_X'], GB1['train_Y']
    val_X, val_Y = GB1['val_X'], GB1['val_Y']
    if len(val_X) == 0:
        val_X, val_Y = train_X, train_Y
    
    test_X, test_Y = GB1['test_X'], GB1['test_Y']
    test_names = GB1['test_names']
           
    return train_X[:, 0:n_f], train_Y, val_X[:, 0:n_f], val_Y, val_X[:, 0:n_f], val_Y, test_X[:, 0:n_f], test_names, test_Y,



class LDA:
    
    def __init__(self, n_components):
        self.n_components = n_components
        self.linear_discriminants = None

    def fit(self, X, y):
        n_features = X.shape[1]
        class_labels = np.unique(y)

        # Within class scatter matrix:
        # SW = sum((X_c - mean_X_c)^2 )

        # Between class scatter:
        # SB = sum( n_c * (mean_X_c - mean_overall)^2 )

        mean_overall = np.mean(X, axis=0)
        SW = np.zeros((n_features, n_features))
        SB = np.zeros((n_features, n_features))
        for c in class_labels:
            X_c = X[y == c]
            mean_c = np.mean(X_c, axis=0)
            # (4, n_c) * (n_c, 4) = (4,4) -> transpose
            SW += (X_c - mean_c).T.dot((X_c - mean_c))

            # (4, 1) * (1, 4) = (4,4) -> reshape
            n_c = X_c.shape[0]
            mean_diff = (mean_c - mean_overall).reshape(n_features, 1)
            SB += n_c * (mean_diff).dot(mean_diff.T)

        self.SW = SW
        # Determine SW^-1 * SB
        A = np.linalg.inv(SW).dot(SB)
        # Get eigenvalues and eigenvectors of SW^-1 * SB
        eigenvalues, eigenvectors = np.linalg.eig(A)
        # -> eigenvector v = [:,i] column vector, transpose for easier calculations
        # sort eigenvalues high to low
        eigenvectors = eigenvectors.T
        idxs = np.argsort(abs(eigenvalues))[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]
        # store first n eigenvectors
        self.linear_discriminants = np.array(eigenvectors[0:self.n_components], dtype=np.float32)

    def transform(self, X):
        # project data
        return np.dot(X, self.linear_discriminants.T)

    def predict(self, X, mu):
        x = self.transform(X)
        return x>mu
        
    

def lda_exp(dataset_name):
    print(f'good: {sum(train_Y==1)}')
    print(f'bad: {sum(train_Y==0)}')

    dimen = 1
    lda = LDA(dimen)
    lda.fit(train_X, train_Y)
    if dimen > 1:
        plot_comparison(lda.linear_discriminants, train_X, train_Y, dim_pref=dimen)
    else:
        x = lda.transform(train_X)
        mu1 = []
        X_c = x[train_Y == 0]
        mu1.append(np.mean(X_c, axis=0))
        
        X_c = x[train_Y == 1]
        mu1.append(np.mean(X_c, axis=0)) 
        
        midpoint1 = (mu1[1]+mu1[0])/2
        
        plt.scatter(x, np.zeros(len(x)), c=train_Y)
        db_y = np.arange(-0.5, 0.5, 0.01)

        plt.plot(midpoint1*np.ones(len(db_y)), db_y, '--b')
        
        
        num_correct=[0,0]
        
        X_c = np.array(x[train_Y == 0], dtype = np.float32)
        num_correct[0] += np.sum((X_c < midpoint1)) 
        X_c = np.array(x[train_Y == 1], dtype = np.float32)
        num_correct[1] += np.sum((X_c > midpoint1)) 
        
        acc0 = num_correct[0]/len(x[train_Y == 0])
        acc1 = num_correct[1]/len(x[train_Y == 1])
        acc = np.sum(num_correct)/len(train_Y)
        
        print(f'num examples: {len(train_Y)}')
        print(f'Best Parameter: {lda.linear_discriminants}, Decision Boundary: mu={midpoint1}')
        print(f"Train acc0: {acc0:.2%}, acc1: {acc1:.2%}, overall acc: {acc:.2%}")
        
    num_correct=[0,0]
    val_X_ = lda.transform(val_X)
    X_c = np.array(val_X_[val_Y == 0], dtype = np.float32)
    num_correct[0] += np.sum((X_c < midpoint1)) 
    X_c = np.array(val_X_[val_Y == 1], dtype = np.float32)
    num_correct[1] += np.sum((X_c > midpoint1)) 
    
    val_acc0 = num_correct[0]/len(val_X_[val_Y == 0])
    val_acc1 = num_correct[1]/len(val_X_[val_Y == 1])
    val_acc = np.sum(num_correct)/len(val_Y)
    print(f"Validation acc0: {val_acc0:.2%}, acc1: {val_acc1:.2%}, overall acc: {val_acc:.2%}")
  

    prediction = lda.predict(test_x, midpoint1)
    ret = zip(test_name, prediction)
    for i in range(len(test_name)):
        print(f'name: {test_name[i]}, prediction: {prediction[i]}')
    
    
    rest_prediction = lda.predict(rest_data, midpoint1)
    rest_map = dict(zip(list(np.squeeze(rest_names)), list(np.squeeze(rest_prediction))))
    rest_dict = sorted(rest_map.items(), key = lambda d: d[1], reverse=True)
    
   
            
    rest_pd = pd.DataFrame(data=rest_dict, columns=['resn', 'prediction'])
    rest_pd.to_csv(cfg[dataset_name]['result_save_path'] + 'rest_prediction_pos.csv')
    print(f'Finished saving prediction for unknown data, number of positive predictions {np.sum(np.squeeze(rest_prediction))}')
 
    plt.show()
    plt.savefig(cfg[dataset_name]['fig_save_path']+'visualCompare.png')
    
    
def compare_rank(y_true, y_pred, name):
    true_curve = sorted(np.squeeze(y_true))
    index = np.argsort(np.squeeze(y_true))
    tmp2 = list(np.squeeze(y_pred))
    pred_curve = [tmp2[i] for i in index]
    
    
    plt.plot(true_curve, 'ob', label = 'True label')
    plt.plot(pred_curve, 'xr', label = 'Predicted label')
    plt.legend()
    plt.xlabel('rank')
    plt.ylabel('value')
    plt.title(cfg[dataset_name]['fig_save_path']+name + '_true_vs_pred')
    plt.show()
    plt.savefig(cfg[dataset_name]['fig_save_path']+name + '_true_vs_pred.png')
    plt.close()
       
    
def run_classifier(dataset_name):
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn import tree
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.svm import SVC
    from sklearn.svm import OneClassSVM
    import sklearn.metrics as metrics
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import classification_report
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import roc_curve
    import seaborn as sns
    
    names = [ 'KNN', 'LR', 'RF', 'DT', 'GBDT', 'SVM', 'one_class_SVM']
    classfiers = [
        # MultinomialNB(alpha=0.01),
        KNeighborsClassifier(),
        LogisticRegression(penalty='l2'),
        RandomForestClassifier(n_estimators=8),
        tree.DecisionTreeClassifier(),
        GradientBoostingClassifier(n_estimators=200),
        SVC(kernel='rbf', probability=True),
        OneClassSVM(gamma='scale')
    ]
    def try_different_method(tmp_name, model):
        if tmp_name == 'one_class_SVM':
            model.fit(train_X)
            score = 1
        else:
            model.fit(train_X,train_Y)
            score = model.score(val_X, val_Y) # score为拟合优度，越大，说明x对y的解释程度越高
        result = model.predict(val_X)
        #plt.figure()
        plt.plot(np.arange(len(result)), val_Y,'g-',label='true value')
        plt.plot(np.arange(len(result)),result,'r-',label='predict value')
        plt.title('%s score: %f' % (tmp_name,score))
        plt.legend()
    
    plt.figure(figsize=(20, 20))
    # sns.set_style("white")
    for i in range(0,len(names)):
        ax = plt.subplot(3,3,i+1)
        plt.xlim(0,20) # 这里只选择绘图展示前20个数据的拟合效果，但score是全部验证数据的得分
        try_different_method(names[i],classfiers[i])
    plt.show()
    plt.savefig(cfg[dataset_name]['fig_save_path'] + 'compare_classifier.png')
    plt.close()
    
    # # ! gred search
    from sklearn.model_selection import GridSearchCV
    from sklearn import metrics

    # 构造基于GridSearchCV的结果打印功能
    class grid():
        def __init__(self,model):
            self.model = model

        def grid_get(self,X,y,param_grid):
            grid_search = GridSearchCV(self.model,param_grid,cv=5,scoring='recall')#, scoring=rmsle_scorer, n_jobs=-1) #n_jobs调用所有核并行计算
            grid_search.fit(X,y)
            #这里为了方便比较mean_test_score，同一取负值并开方
            print('Best params is ',grid_search.best_params_, grid_search.best_score_) #打印最优参数
            grid_search.cv_results_['mean_test_score'] = grid_search.cv_results_['mean_test_score']
            pd.set_option('display.max_columns', None) # 显示所有列
            pd.set_option('max_colwidth',100) # 设置列的显示长度为100，默认为50
            print(pd.DataFrame(grid_search.cv_results_)[['params','mean_test_score','std_test_score']]) #打印所有结果

    #以随机森林为例进行调参，可更改为其他方法，变更grid()内函数即可
    #这里的np.log1p(y_train)是为了保证y_train值尽量贴近正态分布，利于回归拟合，请根据实际情况进行调整
    #同时，np.log1p(y_train)的处理也是为了和rmsle评估函数的功能对应起来，因为该函数默认会先进行log变换回去（convertExp=True）
    # grid(GradientBoostingClassifier()).grid_get(train_X, train_Y,{'random_state':[42], 'n_estimators':[100, 120, 140, 160, 180, 200, 220, 240, 260]}) #随机森林
    grid(OneClassSVM()).grid_get(train_X, train_Y, { 'kernel':['linear', 'poly', 'rbf', 'sigmoid']}) #随机森林
    # import pdb; pdb.set_trace()
    
    
    # * chose one classifier
    regr = GradientBoostingClassifier(n_estimators=200)
    # regr =  SVC(kernel='rbf', probability=True, class_weight='balanced')
    regr.fit(train_X, train_Y)
    # regr = OneClassSVM(gamma='scale', kernel='rbf')
    # regr.fit(train_X[train_Y==1, :])
    
   
    # The mean squared error
    target_names = ['budingwei', 'dingwei']
    support = classification_report(train_Y, regr.predict(train_X), target_names=target_names)
    print(support)
    
    accuracy = accuracy_score(train_Y, regr.predict(train_X))
    print("Train accuracy:", accuracy)
    compare_rank(train_Y, regr.predict(train_X), name='GB_train')
    print("="*30)
    
    support = classification_report(val_Y, regr.predict(val_X), target_names=target_names)
    print(support)
    accuracy = accuracy_score(val_Y, regr.predict(val_X))
    print("Val accuracy:", accuracy)
    compare_rank(val_Y, regr.predict(val_X), name='GB_val')
    print("="*30)
    
    support = classification_report(rest_label, regr.predict(rest_data), target_names=target_names)
    print(support)
    accuracy = accuracy_score(rest_label, regr.predict(rest_data))
    print(f"Test accuracy: {accuracy}")
    compare_rank(rest_label, regr.predict(rest_data), name='GB_test')
    
    
    # filter test
    test_predict = regr.predict(rest_data)  
    print(f'Classified Positive: {np.sum(np.array(test_predict)==1)}')
    plt.figure()
    plt.scatter(range(len(test_predict)), test_predict,label='true value')
    plt.scatter(range(len(rest_label)), rest_label,label='predict value')
    plt.legend()
    plt.show()
    plt.savefig(cfg[dataset_name]['fig_save_path'] + 'agent1_pred_cls.png')
    plt.close()

    
    
    # sort by score
    predict_rank = dict(zip(np.squeeze(rest_names), test_predict))
    sort_predict_rank = sorted(predict_rank.items(), key=lambda d: d[1], reverse=True)
    rest_pd = pd.DataFrame(data=sort_predict_rank, columns=['resn', 'prediction'])
    rest_pd.to_csv(cfg[dataset_name]['result_save_path'] + 'agent1_classification_top1000.csv')
    
    top_freq =np.array([sort_predict_rank[i][1] for i in range(len(sort_predict_rank))])
    # length = min(int(sum(top_freq[top_freq==1])), 1000)
    # top_sort_predict_rank = [sort_predict_rank[i][0] for i in range(length)] # rank top 1000 
    # top_sort_predict_rank_save = pd.DataFrame(top_sort_predict_rank, columns=['resn'])
    # top_sort_predict_rank_save.to_csv(cfg[dataset_name]['result_save_path']+'agent1_classification_top1000.csv')
    

    tmp_freq = np.exp(top_freq)-1
    plt.plot(top_freq)
    plt.show()
    plt.savefig(cfg[dataset_name]['fig_save_path'] + 'agent1_pred_cls.png')
        
        
  
    
    
    
if __name__ == '__main__':
    with open('lda_config.yaml', 'r') as config:
        cfg = yaml.safe_load(config)
    dataset_name = 'GB1_set1ps_change2ps_consider1ps'
    # 'GB1_score3_consider1ps'
    # fpath = cfg[dataset_name]['dataset_path']
    # rest_fpath = cfg[dataset_name]['testset_path']
    # train_X, train_Y, val_X,  val_Y, test_x, test_name= read_dataset(fpath, shuffle=True, augment=True)
    # rest_data, rest_names = read_rest_data(rest_fpath)
    
    train_X, train_Y, val_X, val_Y, test_x, test_name, rest_data, rest_names, rest_label = load_dataset(dataset_name, cfg)
    train_Y[train_Y==0]=-1
    val_Y[val_Y==0]=-1  
      
    if type(train_Y) != list:
        train_Y= train_Y.astype('int')
    if type(val_Y) != list:
        val_Y=val_Y.astype('int')
    if type(rest_label) != list:
        rest_label=rest_label.astype('int')
    
    if len(test_x)==0:
        test_x = val_X
    tmp_x = np.concatenate((train_X, val_X))
    tmp_x = np.concatenate((tmp_x, test_x))
    tmp_x = np.concatenate((tmp_x, rest_data))

    standard_scaler = preprocessing.StandardScaler()
    tmp_x = standard_scaler.fit_transform(tmp_x)
    train_X = standard_scaler.transform(train_X)
    val_X = standard_scaler.transform(val_X)
    test_x = standard_scaler.transform(test_x)
    rest_data = standard_scaler.transform(rest_data)

    run_classifier(dataset_name)
    # lda_exp(dataset_name)







