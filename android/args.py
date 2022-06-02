import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--config',
                    help='path to config file;',
                    type=str,
                    # required=True,
                    default='default.cfg')
    
    parser.add_argument('--client_id',
                    help='client id, only used for latency sampling',
                    type=int,
                    # required=True,
                    default='0')
    
    return parser.parse_args()
