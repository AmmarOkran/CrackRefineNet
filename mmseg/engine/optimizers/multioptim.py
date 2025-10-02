from collections import OrderedDict
from mmengine.optim import DefaultOptimWrapperConstructor, build_optim_wrapper, OptimWrapperDict
from mmseg.registry import OPTIM_WRAPPER_CONSTRUCTORS

@OPTIM_WRAPPER_CONSTRUCTORS.register_module()
class MultipleOptimizerConstructor:
    def __init__(self, optim_wrapper_cfg, **kwargs):
        self.optim_wrapper_cfg = optim_wrapper_cfg

    def __call__(self, model):
        
        optimizers = self.optim_wrapper_cfg["optimizers"]

        optim_wrappers = OrderedDict()
        for key, optim_cfg in optimizers.items():
            module = getattr(model, key)
            optim = build_optim_wrapper(module, optim_cfg)
            optim_wrappers[key] = optim

        return OptimWrapperDict(**optim_wrappers)