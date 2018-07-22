import random
import pandas as pd
import numpy as np
from models import gbt
from utils import load_energy
from setting import DATA_SET_DIR, GBT_RES_DIR, FIG_DIR
from sklearn.metrics import mean_squared_error
from metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt


def agg_all_sum(freqs):
    apts = range(1, 115)
    print('# apartments:', len(apts))
    print('freqs:', freqs)

    agg_energy = {}
    for apt in apts:
        print('reading %d ...' % apt)
        agg_energy[apt] = load_energy(apt)
    df = pd.DataFrame(agg_energy)

    for freq in freqs:
        print('freq:', freq)
        df_freq = df.resample(freq).mean()
        df_freq = df_freq.loc[pd.date_range(start='2016-01-01', end='2016-12-01', freq=freq)]

        df_freq['sum'] = df_freq.apply(lambda x: np.sum(x), axis=1)

        df_freq = df_freq['sum']

        filename = DATA_SET_DIR + 'SUM_%d_%s_2016.pkl' % (len(apts), freq)
        print('saving to file: %s ...' % filename)
        df_freq.to_pickle(filename)
        print('saved.')


def agg_data(n_apt, seed):
    # random.seed(seed)
    # split = 28
    # apts = range(1, split) + range(split+1, 115)
    # random.shuffle(apts)
    # apts = apts[:n_apt-1]
    # apts.append(split)

    random.seed(seed)
    apts = range(1, 115)
    random.shuffle(apts)
    apts = apts[:n_apt]

    print('num of apts:', len(apts))
    agg_energy = {}
    for apt in apts:
        print('reading %d ...' % apt)
        agg_energy[apt] = load_energy(apt)

    df = pd.DataFrame(agg_energy)

    def agg_mean(row):
        return np.mean(row)

    df['mean'] = df.apply(agg_mean, axis=1)

    df = df['mean']

    filename = DATA_SET_DIR + 'Mean_seed_%d_apt_%d_2016.pkl' % (seed, n_apt)
    print('saving to file: %s ...' % filename)
    df.to_pickle(filename)
    print('saved.')


def run(n_apt, seed, freq):
    season = {
        'spring': {'trb': '2016-04-01', 'tre': '2016-04-28', 'teb': '2016-04-29', 'tee': '2016-04-30'},
        'summer': {'trb': '2016-07-01', 'tre': '2016-07-28', 'teb': '2016-07-29', 'tee': '2016-07-30'},
        'autumn': {'trb': '2016-09-01', 'tre': '2016-09-28', 'teb': '2016-09-29', 'tee': '2016-09-30'},
        'winter': {'trb': '2016-11-01', 'tre': '2016-11-28', 'teb': '2016-11-29', 'tee': '2016-11-30'},
    }
    ss = 'summer'
    gbt.run_model(apt_fname=DATA_SET_DIR + 'Mean_seed_%d_apt_%d_2016.pkl' % (seed, n_apt),
                  tr_te_split=season[ss],
                  res_file_pref='%sagg_seed_%d_apt_%s_%s' % (GBT_RES_DIR % freq, seed, n_apt, ss),
                  freq=freq,
                  is_draw=False)


def draw(freq):
    mse_all = []
    mape_all = []
    apts = []

    for filename in ['agg_seed_90_apt_2_summer_mse0.0552_mape0.4141.pkl',
                     'agg_seed_90_apt_4_summer_mse0.0139_mape0.3552.pkl',
                     'agg_seed_90_apt_8_summer_mse0.0295_mape0.3200.pkl',
                     'agg_seed_90_apt_16_summer_mse0.0134_mape0.2330.pkl',
                     'agg_seed_90_apt_32_summer_mse0.0087_mape0.1598.pkl',
                     'agg_seed_90_apt_64_summer_mse0.0071_mape0.1076.pkl',
                     'agg_seed_90_apt_114_summer_mse0.0038_mape0.0935.pkl'
                     ]:

        apt = filename.split('_')[4]
        # ss = filename.split('_')[1].title()

        res = cPickle.load(open(GBT_RES_DIR % freq + filename, 'rb'))

        y_test = res['y_test']
        y_pred = res['y_pred']

        mse = mean_squared_error(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)

        mse_all.append(mse)
        mape_all.append(mape)
        apts.append(apt)

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(mse_all)
    # ax1.set_title('Apartment %s' % apt)
    ax1.set_xlabel('Granularity')
    ax1.set_xticklabels(apts)
    ax1.set_ylabel('MSE')

    ax2.plot(mape_all)
    # ax2.set_title('Apartment %s' % apt)
    ax2.set_xlabel('Granularity')
    ax2.set_xticklabels(apts)
    ax2.set_ylabel('MAPE')

    f.suptitle('Aggregation Performance', fontsize=16)
    plt.savefig(FIG_DIR + 'agg', dpi=300, bbox_inches='tight')

    plt.show()


if __name__ == '__main__':
    # for seed in [12, 34, 42, 56, 90, 113, 128, 100, 7, 29]:
    #     print 'seed:', seed
    #     for i in [2, 4, 8, 16, 32, 64, 114]:
    #         agg_data(i, seed, '1h')
    #         run(i, seed, '1h')
    # draw('1h')

    agg_all_sum(['15T', '30T', '1h', '2h'])



