import logging.config
import json

def log_conf(conf):
    config_dict = json.loads(''.join(open(conf).readlines()))
    logging.config.dictConfig(config_dict)