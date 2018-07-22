from matplotlib import pyplot
from sklearn import svm
from sklearn.metrics import mean_squared_error
from utils import load_data_svm, load_data
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit
from metrics import mean_absolute_percentage_error
import cPickle
from setting import APT_CSV, SVM_RES_DIR


def run_model(apt_fname, tr_te_split, res_file_pref, freq, is_draw):
    ############
    # load data
    ############
    print '========================================' * 2
    print apt_fname, res_file_pref
    print tr_te_split

    df = load_data(apt_fname, freq)
    train = df[tr_te_split['trb']: tr_te_split['tre']]
    test = df[tr_te_split['teb']: tr_te_split['tee']]
    print test
    print 'train/test:', train.shape, test.shape

    feat = list(train.columns.values)
    feat.remove('energy')
    feat.remove('raw_energy')
    print 'raw features (%d):' % len(feat), feat

    X_train = train[feat].as_matrix()
    y_train = train['energy'].as_matrix()
    X_test = test[feat].as_matrix()
    y_test = test['raw_energy'].as_matrix()

    print 'train/test (after converting to matrix):', X_train.shape, X_test.shape

    ####################
    # feature selection
    ####################
    # print 'feature seleciton ...'
    # selected = feature_selection(X_train, y_train, 12)
    # print len(selected)
    # print'selected features (%d):' % sum(selected), [feat[i] for i in range(len(selected)) if selected[i]]
    # X_train = X_train[:, selected]
    # X_test = X_test[:, selected]
    # print 'train/test (after feature selection):', X_train.shape, X_test.shape

    ########
    # train
    ########
    print 'training ...'
    parameters = {'C': (0.001, 0.01, 0.1, 1),
                  'kernel': ['rbf', 'linear', 'poly', 'sigmoid']
                  }

    clf = GridSearchCV(svm.SVR(),
                       param_grid=parameters,
                       cv=TimeSeriesSplit(n_splits=3),
                       scoring='neg_mean_squared_error')
    clf.fit(X_train, y_train)
    print clf.best_params_

    #######
    # test
    #######
    print 'testing ...'

    y_pred = clf.predict(X_test)

    # y_pred = np.exp(np.cumsum(np.concatenate(([np.log(y_test[0])], y_pred))))
    y_pred = np.exp(y_pred)

    #############
    # evaluation
    #############
    mse = mean_squared_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    print 'MSE:', mse
    print 'MAPE:', mape
    print 'save result to file ...'
    cPickle.dump(
        {'y_test': y_test, 'y_pred': y_pred},
        open(res_file_pref + '_mse%.4f_mape%.4f.pkl' % (mse, mape), 'wb'))
    print 'saved.'

    if is_draw:
        pyplot.plot(y_test)
        pyplot.plot(y_pred, color='red')
        pyplot.show()


if __name__ == '__main__':
    season = {
        'spring': {'trb': '2016-04-01', 'tre': '2016-04-28', 'teb': '2016-04-29', 'tee': '2016-04-30'},
        'summer': {'trb': '2016-07-01', 'tre': '2016-07-28', 'teb': '2016-07-29', 'tee': '2016-07-30'},
        'autumn': {'trb': '2016-09-01', 'tre': '2016-09-28', 'teb': '2016-09-29', 'tee': '2016-09-30'},
        'winter': {'trb': '2016-11-01', 'tre': '2016-11-28', 'teb': '2016-11-29', 'tee': '2016-11-30'},
    }

    run_model(apt_fname=APT_CSV % 54,
              tr_te_split=season['spring'],
              res_file_pref='%sapt%d_%s' % (SVM_RES_DIR, 54, 'spring'))
