import cPickle
import numpy as np
import pandas as pd
from setting import GBT_RES_DIR, SVM_RES_DIR, LSTM_RES_DIR, FIG_DIR
import matplotlib.pyplot as plt
import os
from pprint import pprint
from sklearn.metrics import mean_squared_error
from metrics import mean_absolute_percentage_error


def gbt_sample():
    mse_dict = {}
    for filename in os.listdir(GBT_RES_DIR):
        if not filename.startswith('apt'):
            continue
        mape = float(filename.split('_')[3][4:-4])
        mse_dict[mape] = filename
    print pprint(sorted(mse_dict.items(), reverse=False)[:50])


def svm_sample():
    mse_dict = {}
    for filename in os.listdir(SVM_RES_DIR):
        if not filename.startswith('apt'):
            continue
        mape = float(filename.split('_')[3][4:-4])
        mse_dict[mape] = filename
    print pprint(sorted(mse_dict.items(), reverse=False)[:50])


def gbt():
    good_res = [(0.0433, 'apt69_summer_mse0.0025_mape0.0433.pkl'),
                (0.0502, 'apt69_winter_mse0.0141_mape0.0502.pkl'),
                (0.0866, 'apt58_winter_mse0.0121_mape0.0866.pkl'),
                (0.0878, 'apt69_spring_mse0.0183_mape0.0878.pkl'),  #
                (0.0944, 'apt27_summer_mse0.0002_mape0.0944.pkl'),
                (0.0951, 'apt29_summer_mse0.0001_mape0.0951.pkl'),  #
                (0.1025, 'apt13_autumn_mse0.0141_mape0.1025.pkl'),  #
                (0.1043, 'apt54_winter_mse0.0236_mape0.1043.pkl'),
                (0.1053, 'apt51_summer_mse0.0150_mape0.1053.pkl'),
                (0.1117, 'apt20_winter_mse0.0091_mape0.1117.pkl'),
                (0.1161, 'apt53_winter_mse0.0317_mape0.1161.pkl'),
                (0.1179, 'apt5_winter_mse0.0709_mape0.1179.pkl'),   #
                (0.1193, 'apt40_summer_mse0.0002_mape0.1193.pkl'),  #
                (0.1221, 'apt29_winter_mse0.0317_mape0.1221.pkl'),
                (0.1256, 'apt17_summer_mse0.0125_mape0.1256.pkl'),
                (0.128, 'apt17_autumn_mse0.0001_mape0.1280.pkl'),
                (0.1283, 'apt13_winter_mse0.0454_mape0.1283.pkl'),
                (0.1297, 'apt84_winter_mse0.1013_mape0.1297.pkl'),
                (0.1307, 'apt33_winter_mse0.1033_mape0.1307.pkl'),
                (0.1336, 'apt27_winter_mse0.0684_mape0.1336.pkl'),
                (0.1341, 'apt86_winter_mse0.2219_mape0.1341.pkl'),
                (0.1362, 'apt30_autumn_mse0.0161_mape0.1362.pkl'),
                (0.1383, 'apt79_winter_mse0.1355_mape0.1383.pkl'),
                (0.14, 'apt114_winter_mse0.0645_mape0.1400.pkl'),
                (0.1424, 'apt91_winter_mse0.0611_mape0.1424.pkl'),
                (0.1425, 'apt17_winter_mse0.0330_mape0.1425.pkl'),
                (0.1456, 'apt42_winter_mse0.2183_mape0.1456.pkl'),
                (0.1467, 'apt99_winter_mse0.0755_mape0.1467.pkl'),
                (0.1494, 'apt92_winter_mse0.2337_mape0.1494.pkl'),
                (0.155, 'apt91_summer_mse0.0142_mape0.1550.pkl'),
                (0.1553, 'apt35_winter_mse0.3995_mape0.1553.pkl'),
                (0.1691, 'apt15_spring_mse0.0587_mape0.1691.pkl'),
                (0.1696, 'apt100_autumn_mse0.0198_mape0.1696.pkl'),
                (0.1752, 'apt52_winter_mse0.8163_mape0.1752.pkl'),
                (0.1763, 'apt68_winter_mse0.0562_mape0.1763.pkl'),
                (0.1765, 'apt74_winter_mse0.7118_mape0.1765.pkl'),
                (0.1803, 'apt40_winter_mse0.0225_mape0.1803.pkl'),
                (0.1805, 'apt11_winter_mse0.2875_mape0.1805.pkl'),
                (0.182, 'apt112_winter_mse0.4876_mape0.1820.pkl'),
                (0.1823, 'apt30_summer_mse0.0101_mape0.1823.pkl'),
                (0.1838, 'apt45_summer_mse0.0222_mape0.1838.pkl'),
                (0.1844, 'apt70_winter_mse0.8324_mape0.1844.pkl'),
                (0.1872, 'apt49_winter_mse0.4894_mape0.1872.pkl'),
                (0.1914, 'apt33_summer_mse0.0127_mape0.1914.pkl'),
                (0.1932, 'apt2_winter_mse0.2563_mape0.1932.pkl'),
                (0.195, 'apt94_winter_mse0.0502_mape0.1950.pkl'),
                (0.1976, 'apt47_winter_mse0.1654_mape0.1976.pkl'),
                (0.1992, 'apt48_winter_mse0.9288_mape0.1992.pkl'),
                (0.2002, 'apt57_winter_mse0.5314_mape0.2002.pkl'),
                (0.2004, 'apt56_winter_mse0.2465_mape0.2004.pkl')]

    for _, filename in good_res:
        apt = int(filename.split('_')[0][3:])
        ss = filename.split('_')[1]
        mse = float(filename.split('_')[2][3:])
        mape = float(filename.split('_')[3][4:-4])

        res = cPickle.load(open(GBT_RES_DIR + filename, 'rb'))

        y_test, y_pred = res['y_test'], res['y_pred']

        plt.figure(figsize=(8, 6))
        plt.plot(y_test)
        plt.plot(y_pred, color='red')
        plt.title('Apartment %d (%s)\tmse:%s, mape:%s' % (apt, ss, mse, mape), fontsize=18)
        plt.show()


def svm():
    good_res = [(0.0604, 'apt69_summer_mse0.0037_mape0.0604.pkl'),
                (0.0679, 'apt4_winter_mse0.0295_mape0.0679.pkl'),
                (0.0725, 'apt69_winter_mse0.0242_mape0.0725.pkl'),
                (0.0783, 'apt69_spring_mse0.0148_mape0.0783.pkl'),  #
                (0.0926, 'apt58_winter_mse0.0149_mape0.0926.pkl'),
                (0.0931, 'apt27_summer_mse0.0001_mape0.0931.pkl'),  #
                (0.1039, 'apt29_summer_mse0.0001_mape0.1039.pkl'),  #
                (0.1058, 'apt29_winter_mse0.0225_mape0.1058.pkl'),
                (0.1095, 'apt17_winter_mse0.0149_mape0.1095.pkl'),  #
                (0.1109, 'apt54_winter_mse0.0276_mape0.1109.pkl'),
                (0.1159, 'apt114_winter_mse0.0556_mape0.1159.pkl'),
                (0.1231, 'apt20_winter_mse0.0084_mape0.1231.pkl'),
                (0.1241, 'apt5_winter_mse0.0788_mape0.1241.pkl'),   #
                (0.1242, 'apt13_winter_mse0.0345_mape0.1242.pkl'),
                (0.1257, 'apt84_summer_mse0.0130_mape0.1257.pkl'),
                (0.1285, 'apt79_winter_mse0.1365_mape0.1285.pkl'),
                (0.132, 'apt91_summer_mse0.0145_mape0.1320.pkl'),
                (0.1373, 'apt30_autumn_mse0.0204_mape0.1373.pkl'),
                (0.1379, 'apt40_summer_mse0.0002_mape0.1379.pkl'),
                (0.1383, 'apt84_winter_mse0.1175_mape0.1383.pkl'),
                (0.1412, 'apt92_winter_mse0.1981_mape0.1412.pkl'),
                (0.1453, 'apt86_winter_mse0.2943_mape0.1453.pkl'),
                (0.1456, 'apt27_winter_mse0.0800_mape0.1456.pkl'),
                (0.147, 'apt26_summer_mse0.0001_mape0.1470.pkl'),
                (0.1489, 'apt99_winter_mse0.0713_mape0.1489.pkl'),
                (0.151, 'apt48_winter_mse0.6885_mape0.1510.pkl'),
                (0.1563, 'apt17_summer_mse0.0122_mape0.1563.pkl'),
                (0.1577, 'apt74_winter_mse0.7693_mape0.1577.pkl'),
                (0.1579, 'apt30_summer_mse0.0113_mape0.1579.pkl'),
                (0.159, 'apt42_winter_mse0.2473_mape0.1590.pkl'),
                (0.1596, 'apt39_autumn_mse0.4193_mape0.1596.pkl'),
                (0.1604, 'apt112_winter_mse0.4889_mape0.1604.pkl'),
                (0.1627, 'apt68_winter_mse0.0511_mape0.1627.pkl'),
                (0.1663, 'apt40_winter_mse0.0236_mape0.1663.pkl'),
                (0.1664, 'apt13_autumn_mse0.0256_mape0.1664.pkl'),
                (0.1676, 'apt15_spring_mse0.0531_mape0.1676.pkl'),
                (0.1713, 'apt53_winter_mse0.0672_mape0.1713.pkl'),
                (0.1726, 'apt78_summer_mse0.0135_mape0.1726.pkl'),
                (0.1733, 'apt35_winter_mse0.3696_mape0.1733.pkl'),
                (0.1735, 'apt51_summer_mse0.0166_mape0.1735.pkl'),
                (0.1737, 'apt52_winter_mse0.7623_mape0.1737.pkl'),
                (0.1743, 'apt62_summer_mse0.0073_mape0.1743.pkl'),
                (0.1754, 'apt100_autumn_mse0.0205_mape0.1754.pkl'),
                (0.1779, 'apt91_winter_mse0.1028_mape0.1779.pkl'),
                (0.1808, 'apt28_winter_mse0.1686_mape0.1808.pkl'),
                (0.189, 'apt94_winter_mse0.0536_mape0.1890.pkl'),
                (0.1894, 'apt70_winter_mse1.0403_mape0.1894.pkl'),
                (0.1895, 'apt47_winter_mse0.1951_mape0.1895.pkl'),
                (0.1939, 'apt28_spring_mse0.0484_mape0.1939.pkl'),
                (0.1946, 'apt58_autumn_mse0.0003_mape0.1946.pkl')]
    for _, filename in good_res:
        apt = int(filename.split('_')[0][3:])
        ss = filename.split('_')[1]
        mse = float(filename.split('_')[2][3:])
        mape = float(filename.split('_')[3][4:-4])

        res = cPickle.load(open(SVM_RES_DIR % '1h' + filename, 'rb'))
        y_test, y_pred = res['y_test'], res['y_pred']
        plt.plot(y_test)
        plt.plot(y_pred, color='red')
        plt.title('Apartment %d (%s)\tmse:%s, mape:%s' % (apt, ss, mse, mape), fontsize=18)
        plt.show()


def season():
    # load gbt
    gbt_mse, gbt_mape = {}, {}
    for filename in os.listdir(GBT_RES_DIR):
        if not filename.startswith('apt'):
            continue
        apt = int(filename.split('_')[0][3:])
        ss = filename.split('_')[1]
        mse = float(filename.split('_')[2][3:])
        mape = float(filename.split('_')[3][4:-4])
        gbt_mse[(apt, ss)] = mse
        gbt_mape[(apt, ss)] = mape

    # load svm
    svm_mse, svm_mape = {}, {}
    for filename in os.listdir(SVM_RES_DIR):
        if not filename.startswith('apt'):
            continue
        apt = int(filename.split('_')[0][3:])
        ss = filename.split('_')[1]
        mse = float(filename.split('_')[2][3:])
        mape = float(filename.split('_')[3][4:-4])
        svm_mse[(apt, ss)] = mse
        svm_mape[(apt, ss)] = mape

    index = gbt_mse.keys()
    df = pd.DataFrame({'gbt_mse': [gbt_mse[i] for i in index],
                       'gbt_mape': [gbt_mape[i] for i in index],
                       'svm_mse': [svm_mse[i] for i in index],
                       'svm_mape': [svm_mape[i] for i in index]},
                      index=[np.array(zip(*index)[1]), np.array(zip(*index)[0])])
    df_spring = df.loc['spring']
    df_summer = df.loc['summer']
    df_autumn = df.loc['autumn']
    df_winter = df.loc['winter']

    print '\t'.join([str(df_spring['gbt_mse'].mean()), str(df_spring['svm_mse'].mean()),
                     str(df_spring['gbt_mape'].mean()), str(df_spring['svm_mape'].mean())])
    print '\t'.join([str(df_summer['gbt_mse'].mean()), str(df_summer['svm_mse'].mean()),
                     str(df_summer['gbt_mape'].mean()), str(df_summer['svm_mape'].mean())])
    print '\t'.join([str(df_autumn['gbt_mse'].mean()), str(df_autumn['svm_mse'].mean()),
                     str(df_autumn['gbt_mape'].mean()), str(df_autumn['svm_mape'].mean())])
    print '\t'.join([str(df_winter['gbt_mse'].mean()), str(df_winter['svm_mse'].mean()),
                     str(df_winter['gbt_mape'].mean()), str(df_winter['svm_mape'].mean())])


    # df_gbt_good_both = df[(df.gbt_mape < df.svm_mape) & (df.gbt_mse < df.svm_mse)]
    # df_svm_good_both = df[(df.gbt_mape > df.svm_mape) & (df.gbt_mse > df.svm_mse)]
    # df_gbt_mse_good = df[df.gbt_mse < df.svm_mse]
    # df_gbt_mape_good = df[df.gbt_mape < df.svm_mape]
    # print len(df_gbt_good_both)
    # print len(df_svm_good_both)
    # print len(df_gbt_mse_good)
    # print len(df_gbt_mape_good)

    # plt.figure(figsize=(6, 4))
    # plt.boxplot([gbt_mape, svm_mape], labels=['gbt', 'svm'])
    # plt.boxplot([gbt_mse, svm_mse], labels=['gbt', 'svm'])
    # plt.show()


def draw_comparison():

    gbt_filename = 'apt29_summer_mse0.0001_mape0.0951.pkl'
    svm_filename = 'apt29_summer_mse0.0001_mape0.1039.pkl'
    lstm_filename = 'apt29_summer_mse0.00006923_mape0.0894_r0.5248_ly1_189_ly2_169.pkl'

    # gbt_filename = 'apt69_spring_mse0.0183_mape0.0878.pkl'
    # svm_filename = 'apt69_spring_mse0.0148_mape0.0783.pkl'
    # lstm_filename = 'apt69_spring_mse0.01271775_mape0.0711_r0.7201_ly1_183_ly2_175.pkl'

    apt = int(gbt_filename.split('_')[0][3:])
    ss = gbt_filename.split('_')[1].title()

    gbt_res = cPickle.load(open(GBT_RES_DIR % '1h' + gbt_filename, 'rb'))
    svm_res = cPickle.load(open(SVM_RES_DIR % '1h' + svm_filename, 'rb'))
    lstm_res = cPickle.load(open(LSTM_RES_DIR % '1h' + lstm_filename, 'rb'))
    y_test, y_pred_gbt, y_pred_svm, y_pred_lstm = \
        gbt_res['y_test'], gbt_res['y_pred'], svm_res['y_pred'], lstm_res['y_pred']

    gbt_mse = mean_squared_error(y_test, y_pred_gbt)
    gbt_mape = mean_absolute_percentage_error(y_test, y_pred_gbt)
    svm_mse = mean_squared_error(y_test, y_pred_svm)
    svm_mape = mean_absolute_percentage_error(y_test, y_pred_svm)
    lstm_mse = mean_squared_error(y_test, y_pred_lstm)
    lstm_mape = mean_absolute_percentage_error(y_test, y_pred_lstm)

    plt.plot(y_test, color='black', label='Original', linestyle='--')
    plt.plot(y_pred_gbt, color='green', label='GBT')
    plt.plot(y_pred_svm, color='blue', label='SVR')
    plt.plot(y_pred_lstm, color='red', label='PowerLSTM')
    # plt.title('Apartment %d (%s)\nGBT mse:%s, mape:%s\nSVM mse:%s, mape:%s\nLSTM mse:%s, mape:%s'
    #           % (apt, ss, gbt_mse, gbt_mape, svm_mse, svm_mape, lstm_mse, lstm_mape),
    #           fontsize=16)
    print 'gbt_mse:', gbt_mse
    print 'svm_mse:', svm_mse
    print 'lstm_mse:', lstm_mse

    print 'gbt_mape:', gbt_mape
    print 'svm_mape:', svm_mape
    print 'lstm_mape:', lstm_mape

    plt.title('Apartment %d (%s)' % (apt, ss), fontsize=16)
    plt.ylabel('Energy consumption', fontsize=14)
    plt.xlabel('Time (hour)', fontsize=14)
    plt.legend()
    plt.savefig(FIG_DIR + 'season_apt%s_%s' % (apt, ss), dpi=300, bbox_inches='tight')
    plt.show()


def draw_single():
    f = 'apt29_summer_mse0.0002_mape0.1051.pkl'

    apt = f.split('_')[0][3:]
    ss = f.split('_')[1].title()
    freq = '15T'
    label = '15 min'

    res = cPickle.load(open(GBT_RES_DIR % freq + f, 'rb'))

    y_test = res['y_test']
    y_pred = res['y_pred']

    plt.plot(y_test, color='black', linestyle='-', label='Original')
    plt.plot(y_pred, color='magenta', linestyle='--', label=label)
    plt.legend(fontsize=12)
    plt.title('Apartment %s (%s)' % (apt, ss), fontsize=14)
    plt.xlabel('Time (Hour)', fontsize=14)
    plt.ylabel('Energy consumption', fontsize=14)
    plt.savefig(FIG_DIR + 'single_apt%s_%s_%s' % (apt, ss, freq), dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    # gbt()
    # svm()
    # gbt_sample()
    # svm_sample()
    # season()
    draw_comparison()
    # draw_single()
