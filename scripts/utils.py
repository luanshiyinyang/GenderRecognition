import time
import os


def get_logdir(root_path):
    timestamp = time.strftime('%m%d-%H%M', time.localtime())
    log_dir = os.path.join(root_path, '{}'.format(timestamp))
    return log_dir