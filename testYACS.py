from defaults import get_cfg_defaults  # local variable usage pattern, or:
# from config import cfg  # global singleton usage pattern
import argparse
from utils.common import merge_yacs_config, save_model, save_best_model, cp_projects


if __name__ == "__main__":
  # cfg = get_cfg_defaults()
  # # cfg.merge_from_file("experiment.yaml")
  # parser = argparse.ArgumentParser()
  # parser.add_argument('config', help = 'path to config file')
  # parser.add_argument('--local_rank', type=int, default=0)
  # argsKnown,argsUNKown= parser.parse_known_args(['configs/tusimple.yaml','--local_rank','2','DATASET.RAW_IMG_SIZE','[120,54]'])
  # cfg.merge_from_file(argsKnown.config)
  # cfg.merge_from_list(argsUNKown)
  # cfg.freeze()
  args,cfg = merge_yacs_config(overrideOpts=['configs/culane.yaml','--local_rank','2','DATASET.RAW_IMG_SIZE','[120,54]'])
  print(cfg)
