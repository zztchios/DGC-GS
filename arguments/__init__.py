#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if key == "theta_range_deg":
                group.add_argument(f"--{key}", type=int, nargs='+', default=value)
            elif key == "translate_range":
                group.add_argument(f"--{key}", type=float, nargs='+', default=value)
            else:
                if shorthand:
                    if t == bool:
                        group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                    else:
                        group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
                else:
                    if t == bool:
                        group.add_argument("--" + key, default=value, action="store_true")
                    else:
                        group.add_argument("--" + key, default=value, type=t)
            

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self.dataset = "LLFF"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda:0"
        self.eval = False
        self.rand_pcd = False
        self.mvs_pcd = False
        self.save_warp = False
        self.use_grad = False
        self.use_rep = False
        self.n_sparse = -1
        self.n_views = 0
        #--------------------zzt--------------------
        self.theta_range_deg = [   4,     2,     1,     0,    -1,    -2,    -4]
        self.translate_range = [0.01,  0.02,  0.04,  0.08,  0.04,  0.02,  0.01]
        # self.translate_range = [0., 0.,  0.,   0.,  0.,  0., 0.]
        # self.translate_range = [0.01, 0.02,  0.03,   0.04,   0.03,   0.02,  0.01]
        # self.theta_range_deg = [   3,    2,     1,      0]
        # self.translate_range = [0.01, 0.02,  0.03,   0.04]
        self.render = False
        #--------------------zzt--------------------

        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.position_lr_delay_steps = 0
        self.position_lr_start = 0
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        
        self.neural_grid = 5e-3
        self.neural_net = 5e-4
        self.error_tolerance = 0.2
        self.split_opacity_thresh = 0.1
        self.soft_depth_start = 6000
        self.hard_depth_start = 6000

        self.shape_pena = 0.001
        self.scale_pena = 0.001
        self.opa_pena = 0.01

        self.lambda_dssim = 0.04
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0001
        self.prune_threshold = 0.01
        self.sample_pseudo_interval = 10
        self.depth_weight = 0.05
        self.dist_thres = 10.
        # self.densify_grad_threshold = 0.002

        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
