import polars as pl
from typing import List, Dict

from moris.utils import *


def get_cur_station(item: List[Dict[str, str]]) -> str:
    if len(item) == 0:
        return ""
    return item[0]["station_code"]


def get_cur_machine(item: List[Dict[str, bool]]):
    return [x["machine_type"] for x in item]


def get_nbr_stations(item: List[Dict[str, str]]):
    res = [x["station_code"] for x in item]
    return res


class Dataset:
    def __init__(self, filename: str):
        filepath = get_path(DIR.DataDir, filename)
        self.data = load_data(filepath)

    @property
    def df_w_st(self):
        df_worker = pl.DataFrame(data=self.data["worker_list"])
        # 工人当前工位 ["w_code", "cur_st"]
        df_w_st = df_worker \
            .with_columns([
                pl.col("curr_station_list")
                .apply(lambda x: get_cur_station(x), return_dtype=pl.Utf8)
                .alias("cur_st")
            ]) \
            .select(["worker_code", "cur_st"]) \
            .rename({"worker_code": "w_code"})
        return df_w_st

    @property
    def df_w_skill(self):
        # 工人技能 ["w_code", "op_code", "op_cat", "e"]
        data = [w["operation_skill_list"] for w in self.data["worker_list"]]
        dfs = [pl.DataFrame(data=d) for d in data]
        df_w_skill = pl.concat(dfs) \
            .rename({"worker_code": "w_code",
                     "operation_code": "op_code",
                     "operation_category": "op_cat",
                     "efficiency": "e"})
        return df_w_skill

    @property
    def df_w_cat(self):
        # 工人技能种类 ["w_code", "op_cat", "e"]
        data = [w["operation_category_skill_list"] for w in self.data["worker_list"]]
        dfs = [pl.DataFrame(data=d) for d in data if d != []]
        df_w_cat = pl.concat(dfs) \
            .rename({"worker_code": "w_code",
                     "operation_category": "op_cat",
                     "efficiency": "e"})
        return df_w_cat

    @property
    def df_worker(self):
        # 合并所有工人信息 ["w_code", "op_code", "op_cat", "e", "cur_st"]
        df = self.df_w_skill \
            .join(self.df_w_st, on=["w_code"], how="left")
        return df

    @property
    def df_process(self):
        df = pl.DataFrame(data=self.data["process_list"]) \
            .rename({"operation": "op_code",
                     "operation_number": "op_id",
                     "operation_category": "op_cat",
                     "machine_type": "m_type",
                     "machine_type_2": "m_type2",
                     "standard_oper_time": "op_time",
                     "fixed_station_code": "fixed_st_code",
                     "fixed_worker_code": "fixed_w_code"})
        return df

    @property
    def df_machine(self):
        df = pl.DataFrame(data=self.data["machine_list"]) \
            .rename({"machine_type": "m_type",
                     "is_machine_needed": "need_m"})
        return df

    @property
    def df_station(self):
        df = pl.DataFrame(data=self.data["station_list"]) \
            .rename({"station_code": "st_code",
                     "line_number": "line_id",
                     "curr_machine_list": "cur_m_list",
                     "neighbor_station_list": "nbr_st_list"}) \
            .with_columns([
                pl.col("cur_m_list")
                .apply(lambda x: get_cur_machine(x), return_dtype=pl.List(pl.Utf8))
                .alias("cur_m_list"),
                pl.col("nbr_st_list")
                .apply(lambda x: get_nbr_stations(x), return_dtype=pl.List(pl.Utf8))
                .alias("nbr_st_list")
            ])
        return df

    @property
    def conf(self):
        df = pl.DataFrame(data=self.data["config_param"]) \
            .rename({"max_worker_per_oper": "max_w_per_op",
                     "max_station_per_worker": "max_st_per_w",
                     "max_cycle_count": "max_cycle_cnt",
                     "max_revisited_station_count": "max_revisited_st_cnt",
                     "volatility_rate": "vol_rate",
                     "volatility_weight": "vol_w",
                     "upph_weight": "upph_w",
                     "max_machine_per_station": "max_m_per_st",
                     "max_station_per_oper": "max_st_per_op"})
        d = df.to_dicts()[0]
        return d

    @property
    def df_joint(self):
        df = pl.DataFrame(data=self.data["joint_operation_list"]) \
            .rename({"joint_operation": "joint_op"})
        return df
