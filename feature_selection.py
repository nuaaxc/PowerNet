import pandas as pd
from sklearn.feature_selection import SelectFromModel
from utils import load_data
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
from matplotlib import pyplot
from setting import APT_CSV
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from metrics import mean_absolute_percentage_error
from setting import GBT_RES_DIR, SVM_RES_DIR, LSTM_RES_DIR, FIG_DIR




def load_X_y():
    apt = 39
    freq = '1h'
    apt_fname = APT_CSV % apt
    df = load_data(apt_fname, freq)

    # train = df['2016-09-01': '2016-11-30']
    train = df['2016-02-01': '2016-11-30']

    print('train/test:', train.shape)

    feat = list(train.columns.values)
    feat.remove('energy')
    feat.remove('raw_energy')
    print('raw features (%d):' % len(feat), feat)

    X_train = train[feat].as_matrix()
    y_train = train['energy'].as_matrix()

    print('train/test (after converting to matrix):', X_train.shape)

    return X_train, y_train, feat


def feature_importance(X, y, feat):
    model = RandomForestRegressor(n_estimators=400, random_state=42)
    model.fit(X, y)

    # show importance scores
    print(feat)
    print(model.feature_importances_)

    ticks = [i for i in range(len(feat))]

    # plot
    pyplot.figure(figsize=(18, 6))
    rect = pyplot.bar(ticks, model.feature_importances_)
    # for r in rect:
    #     height = r.get_height()
    #     pyplot.text(r.get_x() + r.get_width() / 2., 1.05 * height, '%.3f' % height, ha='center', va='bottom')
    pyplot.xticks(ticks, feat, rotation=70)
    pyplot.ylabel('Feature importance', fontsize=16)
    pyplot.show()


def feature_selection_analysis(X, y, feat):

    rfe = RFE(RandomForestRegressor(n_estimators=800, random_state=42), 5)
    fit = rfe.fit(X, y)

    # report selected features
    print('Selected Features:')
    for i in range(len(fit.support_)):
        if fit.support_[i]:
            print(feat[i])

    # plot feature rank
    ticks = [i for i in range(len(feat))]
    pyplot.bar(ticks, fit.ranking_)
    pyplot.show()


def feature_selection(X, y, num_feat):
    rfe = RFE(RandomForestRegressor(n_estimators=500, random_state=42), num_feat)
    fit = rfe.fit(X, y)
    return fit.support_


def draw_comparison():
    org_filename = 'apt39_autumn_mse0.4409_mape0.2328.pkl'
    sel_filename = 'apt39_autumn_feature_mse0.4141_mape0.2014.pkl'

    apt = int(org_filename.split('_')[0][3:])
    ss = org_filename.split('_')[1].title()

    org_res = pickle.load(open(GBT_RES_DIR % '1h' + org_filename, 'rb'))
    sel_res = pickle.load(open(GBT_RES_DIR % '1h' + sel_filename, 'rb'))

    y_test = org_res['y_test']
    y_pred_org = org_res['y_pred']
    y_pred_sel = sel_res['y_pred']

    org_mse = mean_squared_error(y_test, y_pred_org)
    org_mape = mean_absolute_percentage_error(y_test, y_pred_org)
    sel_mse = mean_squared_error(y_test, y_pred_sel)
    sel_mape = mean_absolute_percentage_error(y_test, y_pred_sel)

    plt.plot(y_test, color='black', label='original', linestyle='--')
    plt.plot(y_pred_org, color='red', label='w/o feature selection')
    plt.plot(y_pred_sel, color='blue', label='feature selection')
    # plt.title('Apartment %d (%s)\nGBT mse:%s, mape:%s\nSVM mse:%s, mape:%s\nLSTM mse:%s, mape:%s'
    #           % (apt, ss, gbt_mse, gbt_mape, svm_mse, svm_mape, lstm_mse, lstm_mape),
    #           fontsize=16)
    print('org_mse:', org_mse)
    print('sel_mse:', sel_mse)

    print('org_mape:', org_mape)
    print('sel_mape:', sel_mape)

    plt.title('Apartment %d (%s)' % (apt, ss), fontsize=16)
    plt.ylabel('Energy consumption', fontsize=14)
    plt.xlabel('Time (hour)', fontsize=14)
    plt.legend()
    plt.savefig(FIG_DIR + 'feat_sel_apt%s_%s' % (apt, ss), dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    # feature_importance(*load_X_y())
    # feature_selection(*load_X_y())
    draw_comparison()
