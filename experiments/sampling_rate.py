from setting import GBT_RES_DIR, FIG_DIR
import cPickle
from sklearn.metrics import mean_squared_error
from metrics import mean_absolute_percentage_error
from matplotlib import pyplot as plt
import os
import pandas as pd
import numpy as np


def compare_two_apts():

    file_2h = 'apt_all_114_feb_mse254.4000_mape0.0677.pkl'
    file_1h = 'apt_all_114_feb_mse199.7673_mape0.0649.pkl'
    file_30T = 'apt_all_114_feb_mse130.9824_mape0.0488.pkl'
    file_15T = 'apt_all_114_feb_mse193.2176_mape0.0605.pkl'

    apt = file_1h.split('_')[1].title()
    ss = file_1h.split('_')[3].title()

    res_2h = cPickle.load(open(GBT_RES_DIR % '2h' + file_2h, 'rb'))
    res_1h = cPickle.load(open(GBT_RES_DIR % '1h' + file_1h, 'rb'))
    res_30T = cPickle.load(open(GBT_RES_DIR % '30T' + file_30T, 'rb'))
    res_15T = cPickle.load(open(GBT_RES_DIR % '15T' + file_15T, 'rb'))

    y_test_2h = res_2h['y_test']
    y_pred_2h = res_2h['y_pred']

    y_test_1h = res_1h['y_test']
    y_pred_1h = res_1h['y_pred']

    y_test_30T = res_30T['y_test']
    y_pred_30T = res_30T['y_pred']

    # y_test_30T = y_test_30T[range(0, len(y_test_30T), 2)]
    # y_pred_30T = y_pred_30T[range(0, len(y_pred_30T), 2)]

    y_test_15T = res_15T['y_test']
    y_pred_15T = res_15T['y_pred']

    # y_test_15T = y_test_15T[range(0, len(y_test_15T), 4)]
    # y_pred_15T = y_pred_15T[range(0, len(y_pred_15T), 4)]

    mse_2h = mean_squared_error(y_test_2h, y_pred_2h)
    mape_2h = mean_absolute_percentage_error(y_test_2h, y_pred_2h)

    mse_1h = mean_squared_error(y_test_1h, y_pred_1h)
    mape_1h = mean_absolute_percentage_error(y_test_1h, y_pred_1h)

    mse_30T = mean_squared_error(y_test_30T, y_pred_30T)
    mape_30T = mean_absolute_percentage_error(y_test_30T, y_pred_30T)

    mse_15T = mean_squared_error(y_test_15T, y_pred_15T)
    mape_15T = mean_absolute_percentage_error(y_test_15T, y_pred_15T)

    print '2h:', mse_2h, mape_2h, y_test_2h.shape
    print '1h:', mse_1h, mape_1h, y_test_1h.shape
    print '30T:', mse_30T, mape_30T, y_test_30T.shape
    print '15T:', mse_15T, mape_15T, y_test_15T.shape

    f, (ax0, ax1, ax2, ax3) = plt.subplots(4, 1, figsize=(12, 12), sharex=True, sharey=True)

    f.add_subplot(111, frameon=False)

    ax0.plot(range(0, 192, 8), y_test_2h, color='black', linestyle='--', label='Original')
    ax0.plot(range(0, 192, 8), y_pred_2h, color='blue', label='2 hours')
    # ax0.set_xlabel('Time (hour)')
    # ax0.set_ylabel('Energy consumption')
    ax0.legend(fontsize=12, loc='lower right')

    ax1.plot(range(0, 192, 4), y_test_1h, color='black', linestyle='--', label='Original')
    ax1.plot(range(0, 192, 4), y_pred_1h, color='red', label='1 hour')
    # ax1.set_xlabel('Time (hour)')
    # ax1.set_ylabel('Energy consumption')
    ax1.legend(fontsize=12, loc='lower right')

    ax2.plot(range(0, 192, 2), y_test_30T, color='black', linestyle='--', label='Original')
    ax2.plot(range(0, 192, 2), y_pred_30T, color='green', label='30 min')
    # ax2.set_xlabel('Time (hour)')
    # ax2.set_ylabel('Energy consumption', fontsize=18)
    ax2.legend(fontsize=12, loc='lower right')

    ax3.plot(range(0, 192, 1), y_test_15T, color='black', linestyle='--', label='Original')
    ax3.plot(range(0, 192, 1), y_pred_15T, color='magenta', label='15 min')
    ax3.set_xlabel('Time (hour)', fontsize=18)
    # ax3.set_ylabel('Energy consumption')
    ax3.legend(fontsize=12, loc='lower right')

    plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    plt.title('All apartments (SUM)', fontsize=18)
    plt.ylabel('Energy consumption', fontsize=18)
    plt.xlabel('Time (hour)', fontsize=18)

    plt.savefig(FIG_DIR + 'sr_apt_%s_%s' % (apt, ss), dpi=300, bbox_inches='tight')
    plt.show()


def compare_two_datasets():
    svm_mse, svm_mape = {}, {}
    for filename in os.listdir(SVM_RES_DIR % '30T'):
        if not filename.startswith('apt'):
            continue
        apt = int(filename.split('_')[0][3:])
        ss = filename.split('_')[1]
        res_30T = cPickle.load(open(SVM_RES_DIR % '30T' + filename, 'rb'))
        y_test_30T = res_30T['y_test']
        y_pred_30T = res_30T['y_pred']
        y_test_30T = y_test_30T[range(1, len(y_test_30T), 2)]
        y_pred_30T = y_pred_30T[range(1, len(y_pred_30T), 2)]
        svm_mse[(apt, ss)] = mean_squared_error(y_test_30T, y_pred_30T)
        svm_mape[(apt, ss)] = mean_absolute_percentage_error(y_test_30T, y_pred_30T)

    index = svm_mse.keys()
    df = pd.DataFrame({'svm_mse': [svm_mse[i] for i in index],
                       'svm_mape': [svm_mape[i] for i in index]},
                      index=[np.array(zip(*index)[1]), np.array(zip(*index)[0])])
    df_spring = df.loc['spring']
    df_summer = df.loc['summer']
    df_autumn = df.loc['autumn']
    df_winter = df.loc['winter']

    print '\t'.join([str(df_spring['svm_mse'].mean()), str(df_spring['svm_mape'].mean())])
    print '\t'.join([str(df_summer['svm_mse'].mean()), str(df_summer['svm_mape'].mean())])
    print '\t'.join([str(df_autumn['svm_mse'].mean()), str(df_autumn['svm_mape'].mean())])
    print '\t'.join([str(df_winter['svm_mse'].mean()), str(df_winter['svm_mape'].mean())])


if __name__ == '__main__':
    compare_two_apts()
    # compare_two_datasets()
