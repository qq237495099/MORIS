import polars as pl
import igraph as ig
import networkx as nx
from typing import List, Dict, Tuple

from .dataset import Dataset
from moris.utils import OpStr


class BaseLoader:
    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        # 设备列表
        self.listMachines = self.dataset.df_machine["m_type"].to_list()
        # 工人列表
        self.listWorkers = self.dataset.df_worker["w_code"].unique().to_list()
        self.WkCnt = len(self.listWorkers)

        # 从小到达排序的工位
        self.listStations = self.dataset.df_station \
            .sort(by=["line_id"], descending=False)["st_code"].to_list()
        # 工位数
        self.StCnt = len(self.listStations)
        self.stToIdx = self.calStrToIdx(self.listStations)
        self.idxToSt = {idx: s for s, idx in self.stToIdx.items()}

        # 工件对应的工序
        partToOps = self.df_process \
            .sort(by=["part_code", "op_id"]) \
            .with_columns(
                pl.concat_str(pl.col("m_type"), pl.col("m_type2"), separator=",")
                .alias("m_type")
            ) \
            .with_columns(
                pl.concat_str(pl.col("m_type"), pl.col("op_code"), separator=";").alias("op_code")
            ) \
            .groupby(by=["part_code"], maintain_order=True).agg(pl.col("op_code")) \
            .to_pandas().set_index(["part_code"]).to_dict(orient="dict")["op_code"]
        self.partToOps = {part: [OpStr(op).to_tpl for op in ops_arr] for part, ops_arr in partToOps.items()}
        self.opToIdx = {part: self.calOpIdx(data) for part, data in self.partToOps.items()}
        # 工序对应的工件
        self.opToPart = self.df_process \
            .select(["op_code", "part_code"]) \
            .to_pandas().set_index(["op_code"]).to_dict(orient="dict")["part_code"]

    @staticmethod
    def calStrToIdx(data) -> Dict[str, int]:
        return {s: idx + 1 for idx, s in enumerate(data)}

    @staticmethod
    def calOpIdx(data) -> Dict[str, int]:
        return {tpl: idx + 1 for idx, tpl in enumerate(data)}

    @property
    def wkTimeMap(self) -> Dict[Tuple[str, str], float]:
        """
        工人做不同工序的做工时间
        :return:
        """
        df = self.dataset.df_process \
            .select(["op_code", "op_cat", "op_time"]).unique() \
            .join(self.dataset.df_worker.select(["w_code", "op_cat", "e"]).unique(),
                  on=["op_cat"], how="left") \
            .select(["w_code", "op_code", "op_time", "e"])
        d = df \
            .with_columns([
                (pl.col("op_time") / pl.col("e")).alias("w_time")
            ]) \
            .to_pandas().set_index(["op_code", "w_code"]).to_dict(orient="dict")["w_time"]
        return d

    @property
    def stToNbrSts(self) -> Dict[str, List[str]]:
        """
        工位对应的邻居工位
        :return:
        """
        d = self.dataset.df_station \
            .select(["st_code", "nbr_st_list"]) \
            .to_pandas().set_index(["st_code"]).to_dict(orient="dict")["nbr_st_list"]
        return d

    @property
    def df_process(self) -> pl.DataFrame:
        """
        工序数据，包含part信息
        :return:
        """
        df = self.dataset.df_process \
            .join(self.dataset.df_joint, on=["part_code"], how="left")
        return df

    @property
    def graph(self) -> ig.Graph:
        """
        工序有向图
        :return:
        """
        df = self.df_process \
            .select(["part_code", "joint_op"]).unique() \
            .rename({"part_code": "from"}) \
            .with_columns([
                pl.col("joint_op")
                .apply(lambda x: self.opToPart[x] if x in self.opToPart else "", return_dtype=pl.Utf8)
                .alias("to")
            ]) \
            .filter(~pl.col("to").is_null()) \
            .with_columns([
                pl.concat_str([pl.col("from"), pl.col("to")], separator=",").alias("name")
            ])
        _g = nx.from_pandas_edgelist(df=df, source="from", target="to", edge_attr=True, create_using=nx.DiGraph)
        g = ig.Graph.from_networkx(_g, vertex_attr_hashable="name")
        return g

    @property
    def conf(self) -> Dict[str, float]:
        return self.dataset.conf
