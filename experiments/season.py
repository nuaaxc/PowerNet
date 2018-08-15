from setting import APT_CSV, GBT_RES_DIR, \
    SVM_RES_DIR, LSTM_RES_DIR, DATA_SET_DIR
from models import gbt
from models import svm
from models import PowerNet_me


season = {
    'spring': {'trb': '2016-04-01', 'tre': '2016-04-28', 'teb': '2016-04-29', 'tee': '2016-04-30'},
    'feb': {'trb': '2016-02-01', 'tre': '2016-02-28', 'teb': '2016-02-29', 'tee': '2016-03-01'},
    'summer': {'trb': '2016-07-01', 'tre': '2016-07-28', 'teb': '2016-07-29', 'tee': '2016-07-30'},
    'autumn': {'trb': '2016-09-01', 'tre': '2016-09-28', 'teb': '2016-09-29', 'tee': '2016-09-30'},
    'winter': {'trb': '2016-11-01', 'tre': '2016-11-28', 'teb': '2016-11-29', 'tee': '2016-11-30'},
}


def run_lstm(freq):
    all_apt(PowerNet_me.run_model, LSTM_RES_DIR, freq)


def run_svm(freq):
    all_apt(svm.run_model, SVM_RES_DIR, freq)


def run_gbt(freq, is_single):
    if is_single:
        single_apt(gbt.run_model, GBT_RES_DIR, freq)
    else:
        all_apt(gbt.run_model, GBT_RES_DIR, freq)


def all_apt(run_model, res_dir, freq):
    for ss in ['spring', 'summer', 'autumn', 'winter']:
        tr_te_split = season[ss]
        for i in range(1, 115):
            run_model(apt_fname=APT_CSV % i,
                      tr_te_split=tr_te_split,
                      res_file_pref='%sapt%d_%s' % (res_dir % freq, i, ss),
                      freq=freq,
                      is_draw=False)


def single_apt(run_model, res_dir, freq):
    i = 114
    ss = 'feb'
    tr_te_split = season[ss]

    # apt_fname = APT_CSV % i
    apt_fname = DATA_SET_DIR + 'SUM_%s_%s_2016.pkl' % (i, freq)
    # res_file_pref = '%sapt%d_%s' % (res_dir % freq, i, ss)
    res_file_pref = '%sapt_all_%d_%s' % (res_dir % freq, i, ss)

    run_model(apt_fname=apt_fname,
              tr_te_split=tr_te_split,
              res_file_pref=res_file_pref,
              freq=freq,
              is_feat_select=False,
              is_draw=False)


if __name__ == '__main__':
    run_gbt('2h', is_single=True)
    # run_gbt('1h', is_single=True)
    # run_gbt('15T', is_single=True)
    # run_gbt('30T', is_single=True)
    # run_lstm()

