import pickle
import os
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error

from metrics import mean_absolute_percentage_error
from models import PowerNet_me, gbt
from setting import APT_CSV, LSTM_RES_DIR, GBT_RES_DIR, FIG_DIR, DATA_SET_DIR

season = {
    'feb': {'trb': '2016-02-01', 'tre': '2016-02-28', 'teb': '2016-03-01', 'tee': '2016-03-30'},
    'mar': {'trb': '2016-03-01', 'tre': '2016-03-30', 'teb': '2016-04-01', 'tee': '2016-04-30'},
    'apr': {'trb': '2016-04-01', 'tre': '2016-04-30', 'teb': '2016-05-01', 'tee': '2016-05-30'},
    'may': {'trb': '2016-05-01', 'tre': '2016-05-30', 'teb': '2016-06-01', 'tee': '2016-06-30'},
    'jun': {'trb': '2016-06-01', 'tre': '2016-06-30', 'teb': '2016-07-01', 'tee': '2016-07-30'},
    'jul': {'trb': '2016-07-01', 'tre': '2016-07-30', 'teb': '2016-08-01', 'tee': '2016-08-30'},
    'aug': {'trb': '2016-08-01', 'tre': '2016-08-30', 'teb': '2016-09-01', 'tee': '2016-09-30'},
    'sep': {'trb': '2016-09-01', 'tre': '2016-09-30', 'teb': '2016-10-01', 'tee': '2016-10-30'},
    'oct': {'trb': '2016-10-01', 'tre': '2016-10-30', 'teb': '2016-11-01', 'tee': '2016-11-30'},
}


def run_gbt(is_true):
    freq = '1h'
    for ss in season.keys():
        print(ss)
        if not is_true:
            use_feat = 'estimate'
            res_file_pref = '%sfp_all_%s_%s' % (GBT_RES_DIR % freq, use_feat, ss)
            gbt.run_model_recursive(apt_fname=DATA_SET_DIR + 'SUM_114_%s_2016.pkl' % freq,
                                    tr_te_split=season[ss],
                                    res_file_pref=res_file_pref,
                                    freq=freq,
                                    is_feat_select=False,
                                    is_draw=False)
        else:
            use_feat = 'truth'
            res_file_pref = '%sfp_all_%s_%s' % (GBT_RES_DIR % freq, use_feat, ss)
            gbt.run_model(apt_fname=DATA_SET_DIR + 'SUM_114_%s_2016.pkl' % freq,
                          tr_te_split=season[ss],
                          res_file_pref=res_file_pref,
                          freq=freq,
                          is_feat_select=False,
                          is_draw=False)


def run_lstm():
    apt_name = 29
    layer1 = 189
    layer2 = 169
    # ss = 'spring_long'
    # ss = 'summer_long'
    # ss = 'autumn_long'
    ss = 'winter_long'

    PowerNet_me.run_model(apt_fname=APT_CSV % apt_name,
                          tr_te_split=season[ss],
                          res_file_pref='%sapt%d_%s' % (LSTM_RES_DIR, apt_name, ss),
                          lstm_param={'layer1': layer1, 'layer2': layer2})


def draw_mape():
    """
    draw mape for both ground-truth & estimate in the same figure.
    """
    month = 'jul'
    freq = '1h'
    # for 'fp_all_truth_'
    mape_all_truth = []
    res = pickle.load(open(GBT_RES_DIR % freq + 'fp_all_truth_%s_mse119.2586_mape0.1352.pkl' % month, 'rb'),
                      encoding='latin1')
    # res = pickle.load(open(GBT_RES_DIR % freq + 'fp_all_truth_%s_mse46.8571_mape0.1122.pkl' % month, 'rb'),
    #                   encoding='latin1')

    y_test = res['y_test']
    y_pred = res['y_pred']
    length = len(y_test)
    for i in range(length - 1):
        y_t = y_test[:i + 1]
        y_p = y_pred[:i + 1]
        # y_t = y_test[i:i + 1]
        # y_p = y_pred[i:i + 1]
        mape_all_truth.append(mean_absolute_percentage_error(y_t, y_p))
    mape_all_truth = np.array(mape_all_truth)

    # for 'fp_all_estimate_'
    mape_all_estimate = []
    res = pickle.load(open(GBT_RES_DIR % freq + 'fp_all_estimate_%s_mse268.7807_mape0.2026.pkl' % month, 'rb'),
                      encoding='latin1')
    # res = pickle.load(open(GBT_RES_DIR % freq + 'fp_all_estimate_%s_mse76.9154_mape0.1367.pkl' % month, 'rb'),
    #                   encoding='latin1')

    y_test = res['y_test']
    y_pred = res['y_pred']
    length = len(y_test)
    for i in range(length - 1):
        y_t = y_test[:i + 1]
        y_p = y_pred[:i + 1]
        # y_t = y_test[i:i + 1]
        # y_p = y_pred[i:i + 1]
        mape_all_estimate.append(mean_absolute_percentage_error(y_t, y_p))
    mape_all_estimate = np.array(mape_all_estimate)

    pprint(mape_all_truth)
    pprint(mape_all_estimate)

    plt.figure(figsize=(5, 4))
    plt.plot(mape_all_truth, c='blue', label='Prediction (Use ground-truth)')
    plt.plot(mape_all_estimate, c='red', label='Prediction (Use estimation)')
    plt.xlabel('Time (hour)', fontsize=14)
    plt.ylabel('MAPE', fontsize=16)
    plt.title('All apartments (SUM, August)', fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(FIG_DIR + 'fp_all_mape_%s_%s' % (month, freq), dpi=300, bbox_inches='tight')

    plt.show()


def draw_mse_mape(how_prefix):

    mse_all, mape_all = [], []
    how, apt, ss, freq = None, None, None, None
    for filename in os.listdir(GBT_RES_DIR % '1h'):
        if filename.startswith(how_prefix):
            ss = filename.split('_')[3].title()
            if ss != 'Aug':
                continue
            apt = filename.split('_')[1]
            how = filename.split('_')[2]
            how = 'estimation' if how == 'estimate' else 'ground-truth'
            freq = '1h'

            res = pickle.load(open(GBT_RES_DIR % freq + filename, 'rb'), encoding='latin1')

            y_test = res['y_test']
            y_pred = res['y_pred']

            mse_a, mape_a = [], []
            length = len(y_test)
            # length = int(len(y_test) / 2)
            print(ss)
            for i in range(length-1):
                y_t = y_test[:i + 1]
                y_p = y_pred[:i + 1]
                # y_t = y_test[i:i + 1]
                # y_p = y_pred[i:i + 1]
                mse = mean_squared_error(y_t, y_p)
                mape = mean_absolute_percentage_error(y_t, y_p)
                mse_a.append(mse)
                mape_a.append(mape)
                print('\t%s\t%s\t%s' % (i + 1, mse, mape))
            mse_all = mse_a
            mape_all = mape_a

    mse_all = np.array(mse_all)
    mape_all = np.array(mape_all)

    # mse_all = np.mean(mse_all, axis=0)
    # mape_all = np.mean(mape_all, axis=0)

    print(mse_all)
    print(mape_all)

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(mse_all)
    # ax1.set_title('Apartment %s' % apt)
    ax1.set_xlabel('Time (hour)')
    ax1.set_ylabel('MSE')

    ax2.plot(mape_all)
    # ax2.set_title('Apartment %s' % apt)
    ax2.set_xlabel('Time (hour)')
    ax2.set_ylabel('MAPE')

    f.suptitle('All apartments (SUM, Use %s)' % how, fontsize=16)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=None)

    # plt.savefig(FIG_DIR + 'fp_all_mse_mape_%s_%s_%s' % (how, apt, freq), dpi=300, bbox_inches='tight')

    plt.show()


def draw_pred():
    freq = '1h'
    month = 'aug'

    res_t = pickle.load(open(GBT_RES_DIR % freq + 'fp_all_truth_%s_mse119.2586_mape0.1352.pkl' % month, 'rb'),
                        encoding='latin1')
    res_e = pickle.load(open(GBT_RES_DIR % freq + 'fp_all_estimate_%s_mse268.7807_mape0.2026.pkl' % month, 'rb'),
                        encoding='latin1')

    y_test = res_t['y_test']
    y_pred_t = res_t['y_pred']
    y_pred_e = res_e['y_pred']

    mse_t = mean_squared_error(y_test, y_pred_t)
    mape_t = mean_absolute_percentage_error(y_test, y_pred_t)

    mse_e = mean_squared_error(y_test, y_pred_e)
    mape_e = mean_absolute_percentage_error(y_test, y_pred_e)

    print('mse:', mse_t, mse_e)
    print('mape:', mape_t, mape_e)

    plt.figure(figsize=(5, 4))
    plt.plot(y_test, color='black', linestyle='--', label='Original')
    plt.plot(y_pred_t, color='blue', linestyle='-', label='Prediction (Use ground-truth)')
    plt.plot(y_pred_e, color='red', linestyle='-', label='Prediction (Use estimation)')
    plt.legend(fontsize=8)
    plt.title('All apartments (SUM, August)', fontsize=14)
    plt.xlabel('Time (Hour)', fontsize=14)
    plt.ylabel('Energy consumption', fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.savefig(FIG_DIR + 'fp_pred_%s_%s' % (freq, month), dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    # run_gbt(True)
    # run_gbt(False)

    draw_mape()
    # draw_mse_mape('fp_all_truth_')
    # draw_mse_mape('fp_all_estimate_')
    # draw_pred()
