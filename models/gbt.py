from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit
from utils import load_data
import numpy as np
from matplotlib import pyplot
from metrics import mean_absolute_percentage_error
import pickle
from feature_selection import feature_selection
from setting import APT_CSV, GBT_RES_DIR


def run_model(apt_fname,
              tr_te_split,
              res_file_pref,
              freq,
              is_feat_select,
              is_draw):
    ############
    # load data
    ############
    print('========================================' * 2)
    print(apt_fname, res_file_pref)
    print(tr_te_split)

    df = load_data(apt_fname, freq)

    train = df[tr_te_split['trb']: tr_te_split['tre']]
    test = df[tr_te_split['teb']: tr_te_split['tee']]
    print(test)
    print('train/test:', train.shape, test.shape)

    feat = list(train.columns.values)
    feat.remove('energy')
    feat.remove('raw_energy')
    print('raw features (%d):' % len(feat), feat)

    X_train = train[feat].as_matrix()
    y_train = train['energy'].as_matrix()
    X_test = test[feat].as_matrix()
    y_test = test['raw_energy'].as_matrix()

    print('train/test (after converting to matrix):', X_train.shape, X_test.shape)

    ####################
    # feature selection
    ####################
    if is_feat_select:
        print('feature seleciton ...')
        selected = feature_selection(X_train, y_train, 12)
        print(len(selected))
        print('selected features (%d):' % sum(selected), [feat[i] for i in range(len(selected)) if selected[i]])
        X_train = X_train[:, selected]
        X_test = X_test[:, selected]
        print('train/test (after feature selection):', X_train.shape, X_test.shape)
        res_file_pref += '_feature'

    ########
    # train
    ########
    print('training ...')
    parameters = {'n_estimators': (50, 100, 150, 200, 250, 300, 350, 400, 450, 500),
                  'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                  'learning_rate': [0.001, 0.01, 0.1],
                  'random_state': [42],
                  'loss': ['ls']}

    clf = GridSearchCV(GradientBoostingRegressor(),
                       param_grid=parameters,
                       cv=TimeSeriesSplit(n_splits=3),
                       scoring='neg_mean_squared_error')
    clf.fit(X_train, y_train)
    print(clf.best_params_)

    #######
    # test
    #######
    print('testing ...')

    y_pred = clf.predict(X_test)

    # y_pred = np.exp(np.cumsum(np.concatenate(([np.log(y_test[0])], y_pred))))
    y_pred = np.exp(y_pred) - 1

    #############
    # evaluation
    #############
    mse = mean_squared_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    print('MSE:', mse)
    print('MAPE:', mape)
    print('save result to file ...')
    pickle.dump(
        {'y_test': y_test, 'y_pred': y_pred},
        open(res_file_pref + '_mse%.4f_mape%.4f.pkl' % (mse, mape), 'wb'))
    print('saved.')

    if is_draw:
        pyplot.plot(y_test)
        pyplot.plot(y_pred, color='red')
        pyplot.show()


def run_model_recursive(apt_fname,
                        tr_te_split,
                        res_file_pref,
                        freq,
                        is_feat_select,
                        is_draw):
    ############
    # load data
    ############
    print('========================================' * 2)
    print(apt_fname, res_file_pref)
    print(tr_te_split)

    df = load_data(apt_fname, freq)
    train = df[tr_te_split['trb']: tr_te_split['tre']]
    test = df[tr_te_split['teb']: tr_te_split['tee']]
    print(test)
    print('train/test:', train.shape, test.shape)

    feat = list(train.columns.values)
    feat.remove('energy')
    feat.remove('raw_energy')
    print('features (%d):' % len(feat), feat)
    print('index of energy-1:', feat.index('energy-1'))

    X_train = train[feat].as_matrix()
    y_train = train['energy'].as_matrix()
    X_test = test[feat].as_matrix()
    y_test = test['raw_energy'].as_matrix()

    print('train/test (after converting to matrix):', X_train.shape, X_test.shape)

    ####################
    # feature selection
    ####################
    if is_feat_select:
        print('feature seleciton ...')
        selected = feature_selection(X_train, y_train, 12)
        print(len(selected))
        print('selected features (%d):' % sum(selected), [feat[i] for i in range(len(selected)) if selected[i]])
        X_train = X_train[:, selected]
        X_test = X_test[:, selected]
        print('train/test (after feature selection):', X_train.shape, X_test.shape)
        res_file_pref += '_feature'

    ########
    # train
    ########
    print('training ...')
    parameters = {'n_estimators': (50, 100, 150, 200, 250, 300, 350, 400, 450, 500),
                  'max_depth': [1, 2, 3],
                  'learning_rate': [0.001, 0.01, 0.1],
                  'random_state': [42],
                  'loss': ['ls']}

    # parameters = {'n_estimators': (50,),
    #               'max_depth': [1],
    #               'learning_rate': [0.001],
    #               'random_state': [42],
    #               'loss': ['ls']}

    clf = GridSearchCV(GradientBoostingRegressor(),
                       param_grid=parameters,
                       cv=TimeSeriesSplit(n_splits=3),
                       scoring='neg_mean_squared_error')
    clf.fit(X_train, y_train)
    print(clf.best_params_)

    #######
    # test
    #######
    print('testing (recursive) ...')
    y_pred = []
    for i in range(len(X_test)):
        # print '-------' * 10
        # print 'i:', i
        # print 'y_pred:', y_pred
        # print 'feat:', X_test[i][18:]
        # print 'range(i):', range(i)
        for j in range(min(i, 49)):
            X_test[i][j+feat.index('energy-1')] = np.log(y_pred[-j-1]+1)
        # print 'feat:', X_test[i][18:]
        y_p = clf.predict([X_test[i]])[0]
        y_p = np.exp(y_p) - 1
        # print 'y_p:', y_p, np.log(y_p+1)
        y_pred.append(y_p)

    #############
    # evaluation
    #############
    mse = mean_squared_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    print('MSE:', mse)
    print('MAPE:', mape)
    print('save result to file ...')
    pickle.dump(
        {'y_test': y_test, 'y_pred': y_pred},
        open(res_file_pref + '_mse%.4f_mape%.4f.pkl' % (mse, mape), 'wb'))
    print('saved.')

    if is_draw:
        pyplot.plot(y_test)
        pyplot.plot(y_pred, color='red')
        pyplot.show()


if __name__ == '__main__':
    season = {
        '2': {'trb': '2016-02-01', 'tre': '2016-02-26', 'teb': '2016-02-27', 'tee': '2016-02-28'},
        '3': {'trb': '2016-03-01', 'tre': '2016-03-28', 'teb': '2016-03-29', 'tee': '2016-03-30'},
        '4': {'trb': '2016-04-01', 'tre': '2016-04-28', 'teb': '2016-04-29', 'tee': '2016-04-30'},
        '5': {'trb': '2016-05-01', 'tre': '2016-05-28', 'teb': '2016-05-29', 'tee': '2016-05-30'},
        '6': {'trb': '2016-06-01', 'tre': '2016-06-28', 'teb': '2016-06-29', 'tee': '2016-06-30'},
        '7': {'trb': '2016-07-01', 'tre': '2016-07-28', 'teb': '2016-07-29', 'tee': '2016-07-30'},
        '8': {'trb': '2016-08-01', 'tre': '2016-08-28', 'teb': '2016-08-29', 'tee': '2016-08-30'},
        '9': {'trb': '2016-09-01', 'tre': '2016-09-28', 'teb': '2016-09-29', 'tee': '2016-09-30'},
        '10': {'trb': '2016-10-01', 'tre': '2016-10-28', 'teb': '2016-10-29', 'tee': '2016-10-30'},
        '11': {'trb': '2016-11-01', 'tre': '2016-11-28', 'teb': '2016-11-29', 'tee': '2016-11-30'},
    }
    freq = '1h'
    apt = 0
    ss = '2'
    run_model(apt_fname=APT_CSV % apt,
              tr_te_split=season[ss],
              res_file_pref='%sapt%d_%s' % (GBT_RES_DIR % freq, apt, ss),
              freq=freq,
              is_feat_select=True,
              is_draw=True)
