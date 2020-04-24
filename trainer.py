## loaded modules ##
import pandas as pd
from os import listdir
from os.path import join,isdir,isfile,exists
import matplotlib.pyplot as plt
import numpy as np
from mlxtend.regressor import StackingCVRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.externals import joblib


##parameters
root_path = './'

in_file = 'NN_dataset.txt'
pred_file = 'predictions.csv'
stat_file = 'statistics.csv'

SEED = 42
np.random.seed(SEED)


def train_func():
    mlp = MLPRegressor(learning_rate='adaptive',
                       hidden_layer_sizes=(23),
                       solver='adam',
                       random_state=1,
                       activation='tanh',
                       max_iter=10000, )

    lasso = Lasso(max_iter=5000, alpha=0.001, random_state=SEED)
    enet = ElasticNet(random_state=SEED, alpha=0.001)
    ridge = Ridge(alpha=1, random_state=SEED)
    rf = RandomForestRegressor(n_estimators=1024,
                               bootstrap=True,
                               max_features='auto',
                               min_samples_leaf=1,
                               min_samples_split=2,
                               random_state=SEED, )
    xgb = GradientBoostingRegressor(random_state=SEED, n_estimators=1024, learning_rate=0.05, )
    stack = StackingCVRegressor(regressors=(ridge, lasso, rf, xgb, enet),
                                meta_regressor=lasso, verbose=1,
                                n_jobs=2, use_features_in_secondary=True)

    df = pd.read_csv(root_path + in_file, index_col='index')


    results = []
    for i in range(13):
        ##preparing input data
        data = df[df['status'] == 'train'].copy()
        data.drop(columns=['STA'], axis=1, inplace=True)
        label_list = [f'label_{i}' for i in range(13)]

        col_list = list(data.columns)
        for lbl in label_list:
            col_list.remove(lbl)
        col_list.remove('status')

        zero_idx = np.isclose(data[col_list].sum(axis=0), 0)
        data.drop(columns=list(data[col_list].columns[zero_idx]), axis=1, inplace=True)

        col_list = list(data.columns)
        for lbl in label_list:
            col_list.remove(lbl)
        col_list.remove('status')

        data[col_list] = data[col_list].apply(lambda x: (x - x.mean()) / (x.std()))

        train, test = train_test_split(data, test_size=0.2, random_state=SEED)
        y_train = train[f'label_{i}']
        label_list.append('status')

        X_train = train.drop(columns=label_list, axis=1)

        #     test.drop(test[test['status'] != 'train'].index, axis=0 , inplace=True)
        y_test = test[f'label_{i}']
        X_test = test.drop(columns=label_list, axis=1)


        ## runing the neural models
        result = {}
        
        for clf, label in zip([ ridge, lasso, rf, xgb, mlp, stack], ['Ridge', 'Lasso',
                                                                     'Random Forest', 'xgb',
                                                                     'mlp', 'StackingClassifier']):
#         for clf, label in zip([stack], ['StackingClassifier']):
            clf.fit(X_train, y_train)
            pred_train = clf.predict(X_train)
            pred_test = clf.predict(X_test)

            mean_test = np.mean(pred_test - y_test)
            std_test = np.std(pred_test - y_test)
            mean_train = np.mean(pred_train - y_train)
            std_train = np.std(pred_train - y_train)

            result['lbl'] = i
            result['mean_test'] = mean_test
            result['mean_train'] = mean_train
            result['std_test'] = std_test
            result['std_train'] = std_train
            result['clf'] = label

            results.append(result.copy())

        joblib.dump(stack, root_path + '{}_{}.pkl'.format(label, i))

    results_pd = pd.DataFrame(results)
    results_pd.to_csv(root_path + stat_file)

def predictor(best_pred = 'StackingClassifier'):
    ## reading the pickle files and add the predictions to it
    df = pd.read_csv(root_path + in_file, index_col='index')
    df.drop(columns=['STA'], axis=1, inplace=True)
    df_temp = df.copy()
    label_list = [f'label_{i}' for i in range(13)]
    label_list.append('status')

    df.drop(columns=label_list, axis=1, inplace=True)
    zero_idx = np.isclose(df.sum(axis=0), 0)
    df.drop(columns=list(df.columns[zero_idx]), axis=1, inplace=True)

    # temp_df = df[(df['status'] == 'train')]
    df = df.apply(lambda x: (x - df_temp[(df_temp['status'] == 'train')][x.name].mean()) / (
        df_temp[(df_temp['status'] == 'train')][x.name].std()))

    temp = {}
    for i in range(13):
        stacking = joblib.load(root_path + '{}_{}.pkl'.format(best_pred, i))
        temp[f'lbl_{i}'] = stacking.predict(df)

    df = df.join(pd.DataFrame(temp))
    df = df.join(df_temp[label_list])
    df.to_csv(root_path + pred_file)

train_func()
predictor()
