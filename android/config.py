import os
import traceback
import sys

DEFAULT_CONFIG_FILE = 'config/default.cfg'

# configuration for FedAvg
class Config():
    def __init__(self, config_file = 'default.cfg'):
        self.config_name = config_file
        self.num_rounds = 1000

        self.fraction_fit=1.0
        self.fraction_eval=0.0
        self.min_fit_clients=1
        self.min_eval_clients=1
        self.min_available_clients=1

        self.ss_baseline=False
        self.fedprox=False
        self.fedbalancer=False
        self.ddl_baseline_fixed=True
        self.ddl_baseline_fixed_value_multiplied_at_mean=1.0
        self.ddl_baseline_smartpc=False
        self.ddl_baseline_wfa=False
        self.num_epochs=5
        self.batch_size=10
        self.clients_per_round = 5
        self.fb_p=0.0
        self.lss=0.0
        self.dss=0.0
        self.w=0
        self.total_client_num=21

        self.output_path = ''
        
        self.read_config(config_file)
        self.log_config()
        
    def read_config(self, filename = DEFAULT_CONFIG_FILE):
        if not os.path.exists(filename):
            assert False
        with open(filename, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                try:
                    line = line.strip().split()
                    if line[0] == 'num_rounds':
                        self.num_rounds = int(line[1])
                    elif line[0] == 'fraction_fit':
                        self.fraction_fit = float(line[1])
                    elif line[0] == 'fraction_eval':
                        self.fraction_eval = float(line[1])
                    elif line[0] == 'min_fit_clients':
                        self.min_fit_clients = int(line[1])
                    elif line[0] == 'min_eval_clients':
                        self.min_eval_clients = int(line[1])
                    elif line[0] == 'min_available_clients':
                        self.min_available_clients = int(line[1])
                    elif line[0] == 'ss_baseline':
                        self.ss_baseline = line[1].strip() == 'True'
                    elif line[0] == 'fedprox':
                        self.fedprox = line[1].strip() == 'True'
                    elif line[0] == 'fedbalancer':
                        self.fedbalancer = line[1].strip() == 'True'
                    elif line[0] == 'ddl_baseline_smartpc':
                        self.ddl_baseline_smartpc = line[1].strip()=='True'
                    elif line[0] == 'ddl_baseline_wfa':
                        self.ddl_baseline_wfa = line[1].strip()=='True'
                    elif line[0] == 'ddl_baseline_fixed':
                        self.ddl_baseline_fixed = line[1].strip()=='True'
                    elif line[0] == 'ddl_baseline_fixed_value_multiplied_at_mean':
                        self.ddl_baseline_fixed_value_multiplied_at_mean = float(line[1].strip())
                    elif line[0] == 'num_epochs':
                        self.num_epochs = int(line[1])
                    elif line[0] == 'batch_size':
                        self.batch_size = int(line[1])
                    elif line[0] == 'clients_per_round':
                        self.clients_per_round = int(line[1])
                    elif line[0] == 'fb_p':
                        self.fb_p = 1.0 - float(line[1].strip())
                    elif line[0] == 'lss':
                        self.lss = float(line[1].strip())
                    elif line[0] == 'dss':
                        self.dss = float(line[1].strip())
                    elif line[0] == 'w':
                        self.w = float(line[1].strip())
                    elif line[0] == 'total_client_num':
                        self.total_client_num = int(line[1])
                    elif line[0] == 'output_path':
                        self.output_path = str(line[1])
                except Exception as e:
                    traceback.print_exc()
    
    def log_config(self):
        configs = vars(self)
        print('================= Config =================')
        for key in configs.keys():
            print('\t{} = {}'.format(key, configs[key]))
        print('================= ====== =================')
        
