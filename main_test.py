import os
import argparse
import logging
import settings
import data_manager
from policy_learner import PolicyLearner


def chooseModelver(stock_code):
    try:
        print("\n",os.listdir('models/{}'.format(stock_code)),'\n')
        idx = int(input("Select model number using index : "))
        return os.listdir('models/{}'.format(stock_code))[idx]
    except:
        raise settings.UndefinedModel()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--code",type=str,default='015760')
    parser.add_argument("--tax",type=str,default='n')
    parser.add_argument("--bal",type=int,default=10000000)
    parser.add_argument("--reward",type=float,default=.02)

    FLAGs, _ = parser.parse_known_args()
    
    stock_code = FLAGs.code
    tax=FLAGs.tax
    bal=FLAGs.bal
    reward=FLAGs.reward

    if tax=='y':
        tax=True
    else:
        tax=False


    model_ver = chooseModelver(stock_code)[6:-3]

    # 로그 기록
    log_dir = os.path.join(settings.BASE_DIR, 'logs/%s' % stock_code)
    timestr = settings.get_time_str()
    file_handler = logging.FileHandler(filename=os.path.join(
        log_dir, "%s_%s.log" % (stock_code, timestr)), encoding='utf-8')
    stream_handler = logging.StreamHandler()
    file_handler.setLevel(logging.DEBUG)
    stream_handler.setLevel(logging.INFO)
    logging.basicConfig(format="%(message)s",
        handlers=[file_handler, stream_handler], level=logging.DEBUG)

    # 주식 데이터 준비
    chart_data = data_manager.load_chart_data(
        os.path.join(settings.BASE_DIR,
                     'data/chart_data/{}.csv'.format(stock_code)))
    prep_data = data_manager.preprocess(chart_data)
    training_data = data_manager.build_training_data(prep_data)

    # 기간 필터링
    training_data = training_data[(training_data['date'] >= '2020-01-01') &
                                  (training_data['date'] <= '2020-02-28')]
    training_data = training_data.dropna()

    # 차트 데이터 분리
    features_chart_data = ['date', 'open', 'high', 'low', 'close', 'volume']
    chart_data = training_data[features_chart_data]

    # 학습 데이터 분리
    features_training_data = [
        'open_lastclose_ratio', 'high_close_ratio', 'low_close_ratio',
        'close_lastclose_ratio', 'volume_lastvolume_ratio',
        'close_ma5_ratio', 'volume_ma5_ratio',
        'close_ma10_ratio', 'volume_ma10_ratio',
        'close_ma20_ratio', 'volume_ma20_ratio',
        'close_ma60_ratio', 'volume_ma60_ratio',
        'close_ma120_ratio', 'volume_ma120_ratio'
    ]
    training_data = training_data[features_training_data]

    # 비 학습 투자 시뮬레이션 시작
    policy_learner = PolicyLearner(
        stock_code=stock_code, chart_data=chart_data, training_data=training_data,
        min_trading_unit=1, max_trading_unit=3,delayed_reward_threshold=reward,tax=tax)
    policy_learner.trade(balance=bal,
                         model_path=os.path.join(
                             settings.BASE_DIR,
                             'models/{}/model_{}.h5'.format(stock_code, model_ver)))
