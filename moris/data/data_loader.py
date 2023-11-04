import polars as pl
from typing import List, Dict, Tuple

from .dataset import Dataset
from .base_loader import BaseLoader


class DataLoader(BaseLoader):
    def __init__(self, dataset: Dataset):
        super().__init__(dataset)

    @property
    def df(self) -> pl.DataFrame:
        """
        设备类型(m)->工序(p)->工人(w)->设备所在当前工位(s)
        :return:
        """
        # 当前工位-设备的对应关系
        df_tmp = self.dataset.df_station \
            .select(["st_code", "cur_m_list"]) \
            .explode("cur_m_list") \
            .rename({"cur_m_list": "m_type"})
        df_wk = self.dataset.df_worker \
            .select(["w_code", "op_cat", "cur_st"])
        # 添加(工人，设备信息，当前工位)字段
        df = self.dataset.df_process \
            .rename({"fixed_st_code": "fix_st", "fixed_w_code": "fix_w"}) \
            .join(df_wk, on=["op_cat"], how="left") \
            .join(self.dataset.df_machine, on=["m_type"], how="left") \
            .join(df_tmp, on=["m_type"], how="left") \
            .with_columns(
            pl.concat_str(pl.col("m_type"), pl.col("m_type2"), separator=",").alias("m_type")
        ) \
            .select(["m_type", "op_code", "w_code", "cur_st", "fix_st", "fix_w", "is_mono", "is_movable", "need_m", "st_code"])
        return df

    @property
    def fixed_alloc(self) -> List[Tuple[str, str, str, str]]:
        """
        固定分配列表（设备，工序，工人，工位）
        :return:
        """
        df = self.df \
            .filter(
                (pl.col("fix_st") != "")
                & (pl.col("fix_w") != "")
            ) \
            .select(["m_type", "op_code", "fix_st", "fix_w"])
        data = [(row["m_type"], row["op_code"], row["fix_w"], row["fix_st"]) for row in df.iter_rows(named=True)]
        return data

    @property
    def listFixSt(self) -> List[str]:
        """
        固定设备所在工位
        :return:
        """
        listFixedSts = self.df \
            .filter(pl.col("is_movable") == False)["st_code"].unique().to_list()
        return listFixedSts

    @property
    def listMoveSt(self) -> List[str]:
        """
        不带固定设备的工位
        :return:
        """
        return list(set(self.listStations) - set(self.listFixSt))

    @property
    def wkToAvailOps(self) -> Dict[str, List[str]]:
        """
        员工对应的可做工序
        :return:
        """
        d = self.df \
            .select(["w_code", "op_code"]).unique() \
            .groupby("w_code").agg(pl.col("op_code")) \
            .to_pandas().set_index(["w_code"]).to_dict(orient="dict")["op_code"]
        return d

    @property
    def fixStMachPair(self) -> List[Tuple[str, str]]:
        """
        固定于某个工位的设备（s,m）
        :return:
        """
        df = self.df \
            .filter(~pl.col("st_code").is_null()) \
            .select(["st_code", "m_type"]).unique()
        data = [(row["st_code"], row["m_type"]) for row in df.iter_rows(named=True)]
        return data

    @property
    def listMonoMachs(self) -> List[str]:
        """
        独占设备
        :return:
        """
        return self.df \
            .filter(pl.col("is_mono") == True)["m_type"].unique().to_list()

    @property
    def listMoveMonoMachs(self) -> List[str]:
        """
        独占设备（可移动设备）
        :return:
        """
        return self.df \
            .filter(
                (pl.col("is_mono") == True)
                & (pl.col("st_code").is_null())
            )["m_type"].unique().to_list()

    @property
    def opToAvailWks(self) -> Dict[Tuple[str, str], List[str]]:
        """
        工序->可分配工人（(m,op) -> [w1,w2,...]）
        :return:
        """
        df = self.df \
            .select(["m_type", "op_code", "w_code"]).unique() \
            .groupby(["m_type", "op_code"]).agg(pl.col("w_code"))
        d = df \
            .to_pandas().set_index(["m_type", "op_code"]).to_dict(orient="dict")["w_code"]
        return d

    @property
    def wkToAvailSts(self) -> Dict[str, List[str]]:
        """
        工人->可分配工位（w -> [s1,s2,...]）
        :return:
        """
        dictSts = {"1": self.listMoveSt}
        df = self.df \
            .select(["w_code", "st_code"]).unique()
        df1 = df.filter(~pl.col("st_code").is_null())
        df2 = df \
            .filter(pl.col("st_code").is_null()) \
            .fill_null("1") \
            .with_columns(
                pl.col("st_code").map_dict(dictSts)
            ) \
            .explode("st_code")
        d = pl.concat([df1, df2]) \
            .groupby(["w_code"]).agg(pl.col("st_code")) \
            .to_pandas().set_index(["w_code"]).to_dict(orient="dict")["st_code"]
        return d

    @property
    def opToAvailSts(self) -> Dict[Tuple[str, str], List[str]]:
        """
        工序->可分配工位（(m,op) -> [s1,s2,...]）
        :return:
        """
        df = self.df \
            .select(["m_type", "op_code", "st_code"]).unique()
        df1 = df.filter(~pl.col("st_code").is_null())
        dictSts = {"1": self.listMoveSt}
        df2 = df \
            .filter(pl.col("st_code").is_null()) \
            .fill_null("1") \
            .with_columns(
                pl.col("st_code").map_dict(dictSts)
            ) \
            .explode("st_code")
        d = pl.concat([df1, df2]) \
            .groupby(["m_type", "op_code"]).agg(pl.col("st_code")) \
            .to_pandas().set_index(["m_type", "op_code"]).to_dict(orient="dict")["st_code"]
        return d

    @property
    def stToAvailMachs(self) -> Dict[str, List[str]]:
        """
        工位->可分配设备（s -> [m1,m2,...]）
        :return:
        """
        dictSts = {"1": self.listMoveSt}
        df = self.df.select(["st_code", "m_type"]).unique()
        df1 = df.filter(~pl.col("st_code").is_null())
        df2 = df \
            .filter(pl.col("st_code").is_null()) \
            .fill_null("1") \
            .with_columns(
                pl.col("st_code").map_dict(dictSts)
            ) \
            .explode("st_code")
        d = pl.concat([df1, df2]) \
            .groupby(["st_code"]).agg(pl.col("m_type")) \
            .to_pandas().set_index(["st_code"]).to_dict(orient="dict")["m_type"]
        return d
