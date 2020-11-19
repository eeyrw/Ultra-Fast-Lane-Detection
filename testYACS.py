from defaults import get_cfg_defaults  # local variable usage pattern, or:
# from config import cfg  # global singleton usage pattern


if __name__ == "__main__":
  cfg = get_cfg_defaults()
  # cfg.merge_from_file("experiment.yaml")
  cfg.freeze()
  for key in cfg.keys():
    print(key)
    print(cfg[key])
