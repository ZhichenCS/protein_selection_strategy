import pandas as pd
import numpy as np
from sklearn import gaussian_process
from sklearn.manifold import TSNE
from sklearn import preprocessing
import matplotlib.pyplot as plt
import yaml

from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import RationalQuadratic
from sklearn.gaussian_process.kernels import WhiteKernel
from sklearn.gaussian_process import GaussianProcessRegressor

from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
#交叉验证和数据集分割函数
from sklearn.model_selection import KFold, cross_val_score,train_test_split 
#各种回归方法
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import ExtraTreeRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.pipeline import make_pipeline
# from lightgbm import LGBMRegressor

import os
os.chdir(os.path.dirname(__file__))

np.random.seed(0)

## todo: budingwei dataset in prediction. rest 8k data in prediction

def load_dataset(dataset_name, cfg):
    if dataset_name == 'PTS1':
        train_X, train_Y, train_fitness, test_x, test_name, test_Y, test_fitness = read_dataset_PTS1(cfg)
    elif dataset_name == 'GB1':
        train_X, train_Y, train_fitness, test_x, test_name, test_Y, test_fitness = read_dataset_GB1(cfg)
    elif dataset_name == 'PhoQ':
        train_X, train_Y, train_fitness, test_x, test_name, test_Y, test_fitness= read_dataset_PhoQ(cfg)
    elif dataset_name == 'GB1_score3_consider1ps':
        train_X, train_Y, train_fitness, test_x, test_name, test_Y, test_fitness= read_dataset_GB1_score3_consider1ps(cfg)
    elif dataset_name == 'GB1_set1ps_change2ps_consider1ps':
        train_X, train_Y, train_fitness, test_x, test_name, test_Y, test_fitness= read_dataset_GB1_set1ps_change2ps_consider1ps(cfg)
    return train_X, train_Y, train_fitness,  test_x, test_name, test_Y, test_fitness
    
    
    
def read_dataset_PTS1(cfg):
    
    dingwei_path = cfg[dataset_name]['dingwei_path']
    budingwei_path = cfg[dataset_name]['budingwei_path']
    rest_path = cfg[dataset_name]['rest_path']
    
    train_X, train_Y, train_fitness, val_X, val_Y, test_X, test_name, test_Y, test_fitness = read_dataset(dingwei_path, budingwei_path, rest_path)
    
    return train_X, train_Y, train_fitness, test_X, test_name, test_Y, test_fitness

def read_dataset_GB1(cfg):
    # * input: read p1, p2, p3, p4, f1,f2,f3, f4
    # * output 
    
    n_f = 4
    GB1 = np.load(cfg[dataset_name]['processed_data_path']+'Pre_GB1.npz', allow_pickle=True)
    train_X, train_Y, train_fitness = GB1['train_X'], GB1['train_Y'], GB1['train_fitness']
    val_X, val_Y = GB1['val_X'], GB1['val_Y']
    if len(val_X) == 0:
        val_X, val_Y = train_X, train_Y
    
    test_X, test_Y, test_fitness = GB1['test_X'], GB1['test_Y'], GB1['test_fitness']
    test_names = GB1['test_names']
    
    return train_X[:, n_f:], train_Y, train_fitness, test_X[:, n_f:], test_names, test_Y, test_fitness

def read_dataset_PhoQ(cfg):
    # * input: read p1, p2, p3, p4, f1,f2,f3, f4
    # * output 
    
    n_f = 4
    data = np.load(cfg[dataset_name]['processed_data_path']+'Pre_PhoQ.npz', allow_pickle=True)
    train_X, train_Y, train_fitness = data['train_X'], data['train_Y'], data['train_fitness']
    val_X, val_Y = data['val_X'], data['val_Y']
    if len(val_X) == 0:
        val_X, val_Y = train_X, train_Y
    
    test_X, test_Y, test_fitness = data['test_X'], data['test_Y'], data['test_fitness']
    test_names = data['test_names']
    
    return train_X[:, n_f:], train_Y, train_fitness, test_X[:, n_f:], test_names, test_Y, test_fitness


def read_dataset_GB1_score3_consider1ps(cfg):
    # * input: read p1, p2, p3, p4, f1,f2,f3, f4
    # * output 
    
    n_f = 4
    data = np.load(cfg[dataset_name]['processed_data_path']+'Pre_PhoQ.npz', allow_pickle=True)
    train_X, train_Y, train_fitness = data['train_X'], data['train_Y'], data['train_fitness']
    val_X, val_Y = data['val_X'], data['val_Y']
    if len(val_X) == 0:
        val_X, val_Y = train_X, train_Y
    
    test_X, test_Y, test_fitness = data['test_X'], data['test_Y'], data['test_fitness']
    test_names = data['test_names']
    
    return train_X[:, n_f:], train_Y, train_fitness, test_X[:, n_f:], test_names, test_Y, test_fitness
    
    
def read_dataset_GB1_set1ps_change2ps_consider1ps(cfg):
    # * input: read p1, p2, p3, p4, f1,f2,f3, f4
    # * output 
    
    n_f = 4
    data = np.load(cfg[dataset_name]['processed_data_path']+'Pre_PhoQ.npz', allow_pickle=True)
    train_X, train_Y, train_fitness = data['train_X'], data['train_Y'], data['train_fitness']
    val_X, val_Y = data['val_X'], data['val_Y']
    if len(val_X) == 0:
        val_X, val_Y = train_X, train_Y
    
    test_X, test_Y, test_fitness = data['test_X'], data['test_Y'], data['test_fitness']
    test_names = data['test_names']
    
    return train_X[:, n_f:], train_Y, train_fitness, test_X[:, n_f:], test_names, test_Y, test_fitness

def read_dataset(fpath1, fpath2, rest_path, shuffle=False):
    dingwei = pd.read_csv(fpath1)
    budingwei = pd.read_csv(fpath2)
    rest_8k = pd.read_csv(rest_path)
    
    used_features = ['f1', 'f2', 'f3'] #,'A', 'B', 'C'
    target = ['freq2']
    
    x = np.array(dingwei[used_features])
    x_bd = np.array(budingwei[used_features])
    test_X = np.array(rest_8k[used_features])
    test_name = np.array(rest_8k['resn'])
    
    if 'f1' in used_features:
        x[:,:3] = np.log(x[:,:3]+1)
        x_bd[:, :3] = np.log(x_bd[:, :3]+1)
        test_X[:, :3] = np.log(test_X[:,:3]+1)
    
    standard_scaler = preprocessing.StandardScaler()
    # transductive normalizing
    tmp_x = np.concatenate((x, x_bd))
    tmp_x = np.concatenate((tmp_x, test_X))
    tmp_x = standard_scaler.fit_transform(tmp_x)
    x = standard_scaler.transform(x)
    x_bd = standard_scaler.transform(x_bd)
    test_X = standard_scaler.transform(test_X)
    
    y = np.array(dingwei[target])
    train_fitness = np.log(y+1)
    y = np.ones_like(train_fitness)
    y_bd = np.array(budingwei[target]) 
    train_fitness_bd = np.log(y_bd+1)
    y_bd = np.zeros_like(train_fitness_bd)
    
    # standard_scaler = preprocessing.StandardScaler()
    # tmp_y = np.concatenate((y, y_bd))
    # tmp_y = standard_scaler.fit_transform(tmp_y)
    # y =  standard_scaler.transform(y)
    # y_bd = standard_scaler.transform(y_bd)
    
    x = np.concatenate((x, x_bd), axis=0)
    y = np.concatenate((y, y_bd))
    fitness = np.concatenate((train_fitness, train_fitness_bd))
  
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
    
    test_Y = np.ones_like(test_name)
    test_fitness = test_Y
    return train_X, train_Y, train_fitness, val_X, val_Y, test_X, test_name, test_Y, test_fitness
        
def plot_comparison(X, y, dim_pref=2, fpath='frequence_space.png'):
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

    print("Computing t-SNE embedding")
    tsne = TSNE(n_components=dim_pref, init='pca', random_state=0)
    X_tsne = tsne.fit_transform(X)
        

    fig = plt.figure()
    if X_tsne.shape[1] > 2 and dim_pref == 3:
        ax = fig.add_subplot(121, projection='3d')
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y)
        ax.set_title('Original Data')
        ax = fig.add_subplot(122, projection='3d')
        ax.scatter(X_tsne[:, 0], X_tsne[:, 1], X_tsne[:, 2], c=y)
        ax.set_title('Transformed Data')
    elif X.shape[1] >= 2:
        ax = fig.add_subplot(121)
        ax.scatter(X[:, 0], X[:, 1], c=y)
        ax.set_title('Original Data')
        ax = fig.add_subplot(122)
        ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y)
        ax.set_title('Transformed Data')
    plt.show()
    plt.savefig(fpath)


def compare_rank(y_true, y_pred, name):
    true_curve = sorted(np.squeeze(y_true))
    index = np.argsort(np.squeeze(y_true))
    tmp2 = list(np.squeeze(y_pred))
    pred_curve = [tmp2[i] for i in index]
    
    
    plt.plot(true_curve, 'ob', label = 'True frequency score')
    plt.plot(pred_curve, 'xr', label = 'Predicted frequency score')
    plt.legend()
    plt.xlabel('rank')
    plt.ylabel('value')
    plt.title(name + '_true_vs_pred')
    plt.show()
    plt.savefig(name + '_true_vs_pred.png')
    plt.close()

def run_regressor():
    # * eval a set of regressors and chose one
    #name需要与regressors一一对应
        train_label = train_fitness
        test_label = test_fitness
        names = ['Decision Tree', 'Linear Regression', 'SVR', 'KNN', 'RFR', 'Ada Boost', 
            'Gradient Boost', 'Bagging', 'Extra Tree','Lasso','ENet','KRR']#,'lgb']
        regressors = [
            DecisionTreeRegressor(),
            LinearRegression(),
            SVR(gamma='scale'),
            KNeighborsRegressor(),
            RandomForestRegressor(),
            AdaBoostRegressor(),
            GradientBoostingRegressor(),
            BaggingRegressor(),
            ExtraTreeRegressor(),    
            make_pipeline(RobustScaler(), Lasso()),
            make_pipeline(RobustScaler(), ElasticNet()),
            KernelRidge(),
            # LGBMRegressor(objective='regression')
        ]

        def try_different_method(tmp_name,model):
            model.fit(train_X,train_label)
            score = model.score(val_X, val_Y) # score为拟合优度，越大，说明x对y的解释程度越高
            result = model.predict(val_X)
            plt.plot(np.arange(len(result)), val_Y,'g-',label='true value')
            plt.plot(np.arange(len(result)),result,'r-',label='predict value')
            plt.title('%s score: %f' % (tmp_name,score))
            plt.legend()
        
        plt.figure(figsize=(20, 20))
        # sns.set_style("white")
        for i in range(0,12):
            ax = plt.subplot(4,4,i+1)
            plt.xlim(0,20) # 这里只选择绘图展示前20个数据的拟合效果，但score是全部验证数据的得分
            try_different_method(names[i],regressors[i])
        plt.show()
        plt.savefig(cfg[dataset_name]['fig_save_path'] + 'agent2_compare_regressor.png')
        plt.close()
        
        regr = GradientBoostingRegressor()
        regr.fit(train_X, train_label)
        
        # The mean squared error
        print("Train MSE: %.2f" % mean_squared_error(train_label, regr.predict(train_X)))
        compare_rank(train_label, regr.predict(train_X), name='GB_train')
        print("Test MSE: %.2f" % mean_squared_error(test_label, regr.predict(test_X)))
        compare_rank(val_Y, regr.predict(val_X), name='GB_val')
        # print(f"Test prediction {regr.predict(test_X)}")
        
        #  * save filtered restult
        test_predict = regr.predict(test_X)
        
        # tmp = list(zip(val_Y, y_predict))
        # for i in tmp: 
        #     print(i[0], i[1])
        
        predict_rank = dict(zip(test_name, test_predict))
        sort_predict_rank = sorted(predict_rank.items(), key=lambda d: d[1], reverse=True)
        
        top_sort_predict_rank = [sort_predict_rank[i][0] for i in range(cfg[dataset_name]['length'])] # rank top 1000 
        
        top_sort_predict_rank_save = pd.DataFrame(top_sort_predict_rank, columns=['resn'])
        top_sort_predict_rank_save.to_csv(cfg[dataset_name]['result_save_path'] + 'agent2_regression_top1000.csv')
        
        top_freq =[sort_predict_rank[i][1] for i in range(len(sort_predict_rank))]
        tmp_freq = np.exp(top_freq)-1
        plt.plot(top_freq)
        plt.show()
        plt.savefig(cfg[dataset_name]['fig_save_path']+'agent2_freq_regr.png')
        
        
        # * save intersection
        predictions = pd.read_csv(cfg[dataset_name]['agent1_result'])
        predict_positive = set(predictions[predictions['prediction']==True]['resn'])

        intersection = predict_positive.intersection(set(top_sort_predict_rank))
        tmp = list(intersection)
       
        np.save(cfg[dataset_name]['result_save_path']+'agent12_regression_intersection', tmp)
        
        true_positive = set(test_name[test_Y==1])
        print(f"Num true positive: {len(true_positive)}, \
              \nRegression recall: {len(true_positive.intersection(set(top_sort_predict_rank)))/len(true_positive)},\
                precision: {len(true_positive.intersection(set(top_sort_predict_rank)))/len(top_sort_predict_rank)}")
        print(f"After filtering, {len(intersection)} remains positive.")# They are {sorted(intersection)}")
        print(f"After filtering, \
            recall: {len(true_positive.intersection(set(intersection)))/len(true_positive)},\
                precision: {len(true_positive.intersection(set(intersection)))/len(intersection)}")
    

def run_classifier():
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn import tree
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.svm import SVC
    import sklearn.metrics as metrics
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import classification_report
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import roc_curve
    import seaborn as sns
    
    if cfg[dataset_name]['cv']:
        # 1. compare base models
        names = [ 'KNN', 'LR', 'RF', 'DT', 'GBDT', 'SVM']
        classifiers = [
            # MultinomialNB(alpha=0.01),
            KNeighborsClassifier(),
            LogisticRegression(penalty='l2'),
            RandomForestClassifier(n_estimators=8),
            tree.DecisionTreeClassifier(),
            GradientBoostingClassifier(n_estimators=200),
            SVC(kernel='rbf', probability=True)
        ]
        def try_different_method(tmp_name,model):
            
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
        for i in range(0,6):
            ax = plt.subplot(4,4,i+1)
            plt.xlim(0,20) # 这里只选择绘图展示前20个数据的拟合效果，但score是全部验证数据的得分
            try_different_method(names[i],classifiers[i])
        plt.show()
        plt.savefig(cfg[dataset_name]['fig_save_path'] + 'compare_classifier.png')
        plt.close()
        
        
        # # ! gred search
        from sklearn.model_selection import GridSearchCV
        from sklearn import metrics

        #构造rmsle评估方法（注意格式输入时两个参数，真实值在前，预测值在后，默认会进行log变换）
        def rmsle(y_true, y_pred, convertExp=True):
            # Apply exponential transformation function
            if convertExp: #这个当时的例子将y进行了log变换，所以结果会先变换回去，可根据实际情况进行修改
                y_true = np.expm1(y_true)
                y_pred = np.expm1(y_pred)      
            # Convert missing value to zero after log transformation
            log_true = np.nan_to_num(np.array([np.log1p(y) for y in y_true]))
            log_pred = np.nan_to_num(np.array([np.log1p(y) for y in y_pred]))    
            # Compute RMSLE
            output = np.sqrt(np.mean((log_true - log_pred)**2))
            return output
            
        # 构造GridSearchCV评估方法，此处利用上述的rmsle评估作为下述网格交叉验证中的评估方法
        rmsle_scorer = metrics.make_scorer(rmsle, greater_is_better=False)#越小越好

        # 构造基于GridSearchCV的结果打印功能
        class grid():
            def __init__(self,model):
                self.model = model

            def grid_get(self,X,y,param_grid):
                grid_search = GridSearchCV(self.model,param_grid,cv=5)#, scoring=rmsle_scorer, n_jobs=-1) #n_jobs调用所有核并行计算
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
        grid(GradientBoostingClassifier()).grid_get(train_X, train_Y,{'random_state':[42], 'n_estimators':[100, 120, 140, 160, 180, 200, 220, 240, 260]}) #随机森林
    
    
    
    # 3. final result
    regr=GradientBoostingClassifier(n_estimators=200)
    # regr =  SVC(kernel='rbf', probability=True)
    regr.fit(train_X, train_Y)
    y_predict = regr.predict(val_X)
    import pdb; pdb.set_trace()

    # The mean squared error
    target_names = ['budingwei', 'dingwei']
    support = classification_report(train_Y, regr.predict(train_X), target_names=target_names)
    print(support)
    accuracy = accuracy_score(train_Y, regr.predict(train_X))
    print(f"Train accuracy: {accuracy} \n" )
    compare_rank(train_Y, regr.predict(train_X), name='GB_train')
    print("="*30)
    
    # support = classification_report(val_Y, regr.predict(val_X), target_names=target_names)
    # print(support)
    # accuracy = accuracy_score(val_Y, regr.predict(val_X))
    # print("Val accuracy:", accuracy)
    # compare_rank(val_Y, regr.predict(val_X), name='GB_val')
    # print("="*30)

    
    testy=  np.expand_dims(test_Y, axis=1)
    support = classification_report(testy, regr.predict(test_X), target_names=target_names)
    print(support)
    accuracy = accuracy_score(test_Y, regr.predict(test_X))
    print(f"Test accuracy: {accuracy} \n")
    compare_rank(test_Y, regr.predict(test_X), name='GB_test')
    
    
    # compare_rank(val_Y, regr.predict(val_X), name='GB_val')
    # accuracy = accuracy_score(y_bd, regr.predict(x_bd))
    # print("bd accuracy:", accuracy)
    
    # compare_rank(y_bd, regr.predict(x_bd), name='GB_bd')
    # print(f"Test prediction {regr.predict(test_X)}")
    
    test_predict = regr.predict(test_X)
    
    # tmp = list(zip(val_Y, y_predict))
    # for i in tmp: 
    #     print(i[0], i[1])
        
    # import pdb; pdb.set_trace()
    # filter test
        
    predict_rank = dict(zip(test_name, test_predict))
    sort_predict_rank = sorted(predict_rank.items(), key=lambda d: d[1], reverse=True)
    top_freq =np.array([sort_predict_rank[i][1] for i in range(len(predict_rank))])
    length = min(int(sum(top_freq[top_freq==1])), cfg[dataset_name]['length'])
    
    top_sort_predict_rank = [sort_predict_rank[i][0] for i in range(length)] # rank top 1000 
    
    top_sort_predict_rank_save = pd.DataFrame(top_sort_predict_rank)
    top_sort_predict_rank_save.to_csv(cfg[dataset_name]['result_save_path']+'classification_top1000.csv')
    

    tmp_freq = np.exp(top_freq)-1
    print(f'Classified Positive: {int(sum(top_freq[top_freq==1]))}')
    plt.plot(top_freq)
    plt.show()
    plt.savefig(cfg[dataset_name]['fig_save_path'] + 'freq_pred_cls.png')
    
    true_positive = set(test_name[test_Y==1])
    print(f"Num true positive: {len(true_positive)} \n \
            Classification recall: {len(true_positive.intersection(set(top_sort_predict_rank)))/len(true_positive)},\
            precision: {len(true_positive.intersection(set(top_sort_predict_rank)))/len(top_sort_predict_rank)}")
    
    #  * save intersection
    predictions =  pd.read_csv(cfg[dataset_name]['agent1_result'])
    predict_positive = set(predictions[predictions['prediction']==True]['resn'])
    intersection = predict_positive.intersection(set(top_sort_predict_rank))
    print(f"After classification filtering, {len(intersection)} out of {len(predict_positive)} remains positive.")# They are {sorted(intersection)}")
    
    print(f"recall: {len(true_positive.intersection(set(intersection)))/len(true_positive)},\
            precision: {len(true_positive.intersection(set(intersection)))/len(intersection)}")

    # tmp = set(pd.read_csv('../../data/regression_top1000.csv', header=0)['0'])
    # print(f'len of regression {len(tmp)}, cls and regr same samples: {len(tmp.intersection(set(top_sort_predict_rank)))}')
    # tmp = ['AAI', 'ACI', 'ACV', 'ADL', 'AEI', 'AEL', 'AEM', 'AFM', 'AGI', 'AGL', 'AGM', 'AHI', 'AKA', 'AKH', 'AKK', 'AKN', 'AKT', 'ALF', 'ALI', 'ALM', 'ALV', 'AMM', 'ANF', 'ANV', 'ANY', 'AQI', 'AQM', 'ARN', 'ARV', 'ASF', 'ASY', 'ATM', 'AVI', 'AVM', 'AWI', 'AWL', 'AWM', 'AYF', 'AYI', 'AYL', 'CAI', 'CAL', 'CAM', 'CCI', 'CCL', 'CCM', 'CFL', 'CGI', 'CGL', 'CGM', 'CHL', 'CKN', 'CKV', 'CKY', 'CLM', 'CMI', 'CMM', 'CNI', 'CNM', 'CQL', 'CQM', 'CRF', 'CRI', 'CRY', 'CSI', 'CSM', 'CTI', 'CTM', 'CWL', 'CYI', 'CYL', 'CYM', 'DCL', 'DFL', 'DKF', 'DKI', 'DKM', 'DKV', 'DKY', 'DLL', 'DML', 'DNL', 'DRI', 'DRL', 'DRM', 'DSL', 'DYL', 'ECL', 'EFL', 'EKF', 'EKI', 'EKM', 'EKV', 'EKY', 'ELL', 'EML', 'ENI', 'ENL', 'ENM', 'ERF', 'ERI', 'ERL', 'ERM', 'ESI', 'ESL', 'ESM', 'EYL', 'FFL', 'FGL', 'FKF', 'FKM', 'FML', 'FQL', 'FRI', 'FRL', 'FRM', 'FSM', 'FTL', 'FYL', 'GCL', 'GGL', 'GHL', 'GKI', 'GKY', 'GLI', 'GLM', 'GML', 'GNL', 'GNM', 'GQL', 'GRI', 'GRM', 'GSI', 'GSM', 'GYL', 'HCL', 'HFL', 'HGL', 'HHL', 'HKF', 'HKI', 'HKM', 'HKY', 'HML', 'HNL', 'HQL', 'HSL', 'HTL', 'HYL', 'ICL', 'IFL', 'IKF', 'IKI', 'IKM', 'IKV', 'IKY', 'ILL', 'IML', 'INL', 'IRI', 'IRL', 'IRM', 'ISL', 'IYL', 'LCL', 'LFL', 'LKF', 'LKI', 'LKM', 'LKV', 'LKY', 'LML', 'LNL', 'LRI', 'LRL', 'LRM', 'LSL', 'LSM', 'LYL', 'MCL', 'MFL', 'MKF', 'MKI', 'MKM', 'MKV', 'MKY', 'MLL', 'MML', 'MNL', 'MRI', 'MRL', 'MRM', 'MSL', 'MYL', 'NAL', 'NCL', 'NGL', 'NHL', 'NKF', 'NKI', 'NKY', 'NLL', 'NML', 'NQL', 'NTL', 'NYL', 'PAI', 'PAM', 'PCI', 'PCL', 'PCM', 'PDL', 'PEI', 'PEL', 'PEM', 'PFM', 'PGI', 'PGL', 'PGM', 'PHL', 'PHM', 'PKN', 'PKY', 'PLF', 'PLM', 'PMI', 'PMM', 'PNF', 'PNI', 'PNM', 'PPL', 'PQI', 'PQL', 'PQM', 'PRN_', 'PRY', 'PSM', 'PTI', 'PTL', 'PTM', 'PVL', 'PWI', 'PWM', 'PYI', 'PYL', 'PYM', 'QAL', 'QCL', 'QFL', 'QGL', 'QHL', 'QKF', 'QKY', 'QLL', 'QML', 'QNL', 'QQL', 'QRI', 'QRM', 'QTL', 'QYL', 'RCL', 'RFL', 'RKF', 'RKI', 'RKM', 'RKV', 'RKY', 'RLL', 'RML', 'RNI', 'RNL', 'RNM', 'RRF', 'RRI', 'RRL', 'RRM', 'RSI', 'RSL', 'RSM', 'RYL', 'SAV', 'SCF', 'SCV', 'SCY', 'SDI', 'SDM', 'SEI', 'SEM', 'SFV', 'SFY', 'SGF', 'SGI', 'SGV', 'SHF', 'SHM', 'SHV', 'SMF', 'SMI', 'SMV', 'SMY', 'SNF', 'SNN', 'SNV', 'SNY', 'SPI', 'SPM', 'SQF', 'SRA', 'SRC', 'SRG', 'SRH', 'SRN', 'SRQ', 'SRS', 'SRT', 'SRW', 'SSN', 'SSY', 'STF', 'SVI', 'SVM', 'SWI', 'SWM', 'SYF', 'SYV', 'SYY', 'TAL', 'TCL', 'TGL', 'THL', 'TKF', 'TKN', 'TKY', 'TLL', 'TML', 'TNM', 'TQL', 'TRY', 'TSI', 'TSM', 'TTL', 'TWL', 'TYL', 'VCL', 'VFL', 'VKF', 'VKI', 'VKM', 'VKV', 'VKY', 'VLL', 'VML', 'VNL', 'VRI', 'VRL', 'VRM', 'VSL', 'VYL', 'WCL', 'WFL', 'WKF', 'WKI', 'WKM', 'WKV', 'WKY', 'WLL', 'WML', 'WNL', 'WRI', 'WRL', 'WRM', 'WSL', 'WYL', 'YCL', 'YFL', 'YKF', 'YKI', 'YKM', 'YKV', 'YKY', 'YLL', 'YML', 'YNL', 'YRI', 'YRL', 'YRM', 'YSL', 'YYL']
    # tmp2 = set(tmp)
    # print(f'compare with old, same samples: {len(tmp2.intersection(set(top_sort_predict_rank)))}')
    
    regresion_prediction = np.load(cfg[dataset_name]['result_save_path']+'agent12_regression_intersection.npy', allow_pickle=True)
    
    regresion_prediction = set(regresion_prediction.tolist())
    intersection = regresion_prediction.intersection(set(top_sort_predict_rank))
    
    
    print(f"After regression & classification filtering {len(regresion_prediction)}, {len(intersection)} remains positive.")# They are {sorted(intersection)}")
    print(f"recall: {len(true_positive.intersection(set(intersection)))/len(true_positive)},\
            precision: {len(true_positive.intersection(set(intersection)))/len(intersection)}")
    
if __name__ == '__main__':
    
    with open('frequency_config.yaml', 'r') as config:
        cfg = yaml.safe_load(config)
    dataset_name = 'GB1_set1ps_change2ps_consider1ps' # 'PhoQ'

    train_X, train_Y, train_fitness, test_X, test_name, test_Y, test_fitness = load_dataset(dataset_name, cfg)
    if type(train_Y) != list:
        train_Y= train_Y.astype('int')
    if type(test_Y) != list:
        rest_label=test_Y.astype('int')
        
    val_X = train_X
    val_Y = train_Y

    tmp_x = np.concatenate((train_X, test_X))

    # plot_comparison(x, y, dim_pref=3, fpath='feature_space.png')
    # plot_comparison(x_bd, y_bd, dim_pref=3, fpath='feature_space_bd.png')
    standard_scaler = preprocessing.StandardScaler()
    tmp_x = standard_scaler.fit_transform(tmp_x)
    train_X = standard_scaler.transform(train_X)
    val_X = standard_scaler.transform(val_X)
    test_x = standard_scaler.transform(test_X)
    print('in regression \n')
    run_regressor()
    print('in classification \n')
    run_classifier()

            
                
    
    
    
            
    
    
            