import os
import sys
import logging
import argparse
import settings
import warnings
import data_manager
from data_loader import DataLoader
from policy_learner import PolicyLearner



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--code",type=str,default='kospi')
    parser.add_argument("--tax",type=str,default='n')
    parser.add_argument("--bal",type=int,default=10000000)
    parser.add_argument("--reward",type=float,default=.02)
    
    FLAGs, _ = parser.parse_known_args()
    
    stock_code = FLAGs.code
    tax=FLAGs.tax
    bal=FLAGs.bal
    reward=FLAGs.reward

    stock = DataLoader(stock_code)

    if bal <= 0:
        raise settings.BalanceUnderflow()

    if stock_code+".csv" not in [file for file in os.listdir('./data/chart_data') if file.endswith(".csv")]:
        stock.makeNewFile()
    else:
        stock.updateFile()

    if tax=='y':
        tax=True
    else:
        tax=False

    log_dir = os.path.join(settings.BASE_DIR, 'logs/%s' % stock_code)
    timestr = settings.get_time_str()
    if not os.path.exists('logs/%s' % stock_code):
        os.makedirs('logs/%s' % stock_code)    
    file_handler = logging.FileHandler(filename=os.path.join(
        log_dir, "%s_%s.log" % (stock_code, timestr)), encoding='utf-8')
    stream_handler = logging.StreamHandler(sys.stdout)
    file_handler.setLevel(logging.DEBUG)
    stream_handler.setLevel(logging.INFO)
    logging.basicConfig(format="%(message)s",
        handlers=[file_handler, stream_handler], level=logging.DEBUG)


    # Prepare the stock data
    chart_data = data_manager.load_chart_data(
        os.path.join(settings.BASE_DIR,
                     'data/chart_data/{}.csv'.format(stock_code)))
    prep_data = data_manager.preprocess(chart_data)
    training_data = data_manager.build_training_data(prep_data)

    # Date range filtering
    training_data = training_data[(training_data['date'] >= '2018-01-01') &
                                  (training_data['date'] <= '2018-12-31')]
    training_data = training_data.dropna()

    # Chart Data Separation
    features_chart_data = ['date', 'open', 'high', 'low', 'close', 'volume']
    chart_data = training_data[features_chart_data]

    # Training data separation
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

    # Strat reinforcement learning
    policy_learner = PolicyLearner(
        stock_code=stock_code, chart_data=chart_data, training_data=training_data,
        min_trading_unit=1, max_trading_unit=2, delayed_reward_threshold=reward, lr=.0001,tax=tax)
    policy_learner.fit(balance=bal, num_epoches=1000,
                       discount_factor=0, start_epsilon=.5)

    # Save Policy Neural Network to File
    model_dir = os.path.join(settings.BASE_DIR, 'models/%s' % stock_code)
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    model_path = os.path.join(model_dir, 'model_%s.h5' % timestr)
    policy_learner.policy_network.save_model(model_path)
