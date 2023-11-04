import os
import json
import time
import platform
from functools import wraps


class OpStr(str):
    @property
    def to_tpl(self):
        return tuple(self.split(";"))


def get_path(*args) -> str:
    return os.path.join(*args)


def get_base_dir() -> str:
    cur_dir = os.path.split(os.path.realpath(__file__))[0]
    if platform.system().lower() == "windows":
        return "\\".join(cur_dir.split("\\")[0:-1])
    else:
        return "/".join(cur_dir.split("/")[0:-1])


class DIR:
    BaseDir = get_base_dir()
    DataDir = get_path(BaseDir, "src", "data", "input")


def load_data(filepath):
    with open(file=filepath, mode="r", encoding="utf-8") as fp:
        data = json.loads(fp.read())
    return data


def time_it(module=None, logger=None):
    def inner(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            s_t = time.time()
            res = func(*args, *kwargs)
            e_t = time.time()
            m = module
            if m is None:
                m = func.__name__
            print("{0} time cost: {1}s".format(m, e_t - s_t))
            if logger is not None:
                logger.info("{0} time cost: {1}s".format(m, e_t - s_t))
            return res
        return wrapper
    return inner
