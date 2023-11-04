import numpy as np
import polars as pl

from .model import Model, tpl_to_str
from moris.data import DataLoader


EPSILON = 1e-6


class OptModel(Model):
    def __init__(self, data_loader: DataLoader):
        super().__init__(data_loader)
        self.var = {}
        self.vard = {}
        self.stToMach = {}
        self.v = {}
        self.vd = {}
        self.dual = {}
        self.tt = {}
        self.obj = []
        self.max_tt = None
        self.allow_help = False

    def allocStToMach(self):
        """
        w变量：(工位-设备)分配关系
        :return:
        """
        # 建立(工位-设备)分配变量
        [self.create_w(s, m) for s, list_m in self.data_loader.stToAvailMachs.items() for m in list_m]
        # 固定设备约束
        for s, m in self.data_loader.fixStMachPair:
            self.AddConstr(self.w[s][m] == 1)
            if m in self.data_loader.listMonoMachs:
                expr = self.Sum([self.w[s][_m] for _m in self.w[s]])
                self.AddConstr(expr == 1)
        # 移动设备中需要考虑独占设备的约束
        for s, list_m in self.w.items():
            for m in list_m:
                # 移动独占设备所分配的工位，不允许其它设备的加入
                if m in self.data_loader.listMoveMonoMachs:
                    # y = (1 - x1) * x2
                    name = "contr_mono_{0}_{1}".format(s, m)
                    u = int(self.data_loader.conf["max_m_per_st"])
                    x1 = 1 - self.w[s][m]
                    x2 = self.Sum([self.w[s][_m] for _m in self.w[s] if _m != m])
                    # 引入新变量y，代表工位s其它设备的入站情况（除去m设备）
                    y = self.IntVar(0, u, name=name)
                    self.AddConstr(y <= u * (1 - x1))
                    self.AddConstr(y <= x2)
                    self.AddConstr(x2 - u * x1 <= y)
                    # 更新s工位的替代变量
                    self.stToMach[s] = self.w[s][m] + y
        # 每个工位的设备数量有上限约束
        for s in self.w:
            if s in self.data_loader.listMoveMonoMachs:
                expr = self.stToMach[s]
            else:
                expr = self.Sum([self.w[s][m] for m in self.w[s]])
            self.AddConstr(expr <= self.data_loader.conf["max_m_per_st"])

    def allocOpToWks(self):
        """
        x变量：(工序-工人)分配关系
        :return:
        """
        # 工序分配到工人
        for op, list_w in self.data_loader.opToAvailWks.items():
            for w in list_w:
                self.create_x(op, w)
        # 每件工序分配的主体人数只能一个(模型不考虑帮扶)
        for op in self.x:
            expr = self.Sum([self.x[op][w] for w in self.x[op]])
            self.AddConstr(expr == 1)

    def allocWkToSts(self):
        """
        y变量：(工人-工位)分配关系
        :return:
        """
        # 工人分配到工位
        [self.create_y(w, s) for w, list_s in self.data_loader.wkToAvailSts.items() for s in list_s]
        for w in self.y:
            expr = self.Sum([self.y[w][s] for s in self.y[w]])
            self.AddConstr(1 <= expr)
            self.AddConstr(expr <= self.data_loader.conf["max_st_per_w"])
        # 每个工位最多分配一个人
        for s in self.data_loader.listStations:
            expr = self.Sum([self.y[w][s] for w in self.y if s in self.y[w]])
            self.AddConstr(expr <= 1)

    def allocOpToSts(self):
        """
        z变量：(工序-工位)分配关系
        :return:
        """
        # 工序分配到工位
        for op, list_s in self.data_loader.opToAvailSts.items():
            for s in list_s:
                self.create_z(op, s)
        # 每件工序分配的站位数只能一个(模型不考虑帮扶)
        for op in self.z:
            expr = self.Sum([self.z[op][s] for s in self.z[op]])
            self.AddConstr(expr == 1)

    def create_var(self):
        """
        var变量：(工序-工人-工位)分配关系
        :return:
        """
        for op, list_w in self.data_loader.opToAvailWks.items():
            self.var.setdefault(op, {})
            for w in list_w:
                self.var[op].setdefault(w, {})
                if op in self.data_loader.opToAvailSts:
                    for s in self.data_loader.opToAvailSts[op]:
                        name = "var_{0}_{1}_{2}".format(tpl_to_str(op), w, s)
                        self.var[op][w][s] = self.BoolVar(name=name)
                else:
                    for s in self.data_loader.listStations:
                        name = "var_{0}_{1}_{2}".format(tpl_to_str(op), w, s)
                        self.var[op][w][s] = self.BoolVar(name=name)

    def addVarConstr(self):
        """
        x,y,z,w,var之间的关系
        :return:
        """
        # 工序，工人确定时，最多只能分配一个工位
        for op in self.x:
            for w in self.var[op]:
                expr = self.Sum([self.var[op][w][s] for s in self.var[op][w]])
                self.AddConstr(expr <= 1)

        # 每个工人做的工件满足上下层级平衡约束
        for w in self.data_loader.listWorkers:
            lhs = self.Sum([self.x[op][w] for op in self.x for _w in self.x[op] if _w == w])
            rhs = self.Sum([self.var[op][w][s] for op in self.var for _w in self.var[op]
                            if _w == w for s in self.var[op][w]])
            self.AddConstr(lhs == rhs)

        # y = z[op][s] * w[s][m]（分配了对应设备的工位，工序才能分配到该工位）
        for op in self.z:
            for s in self.z[op]:
                if s in self.w and op[0] in self.w[s]:
                    x1 = self.w[s][op[0]]
                    x2 = self.z[op][s]
                    # 引入新的变量y，代表工序op是否最终分配到了工位s
                    name = "constr_eq_{0}_{1}".format(tpl_to_str(op), s)
                    y = self.BoolVar(name=name)
                    self.AddConstr(y <= x1)
                    self.AddConstr(y <= x2)
                    self.AddConstr(x1 + x2 - 1 <= y)
                    # 记录新的替代变量，v替代z
                    self.v.setdefault(op, {})
                    self.v[op][s] = y

        # var等效约束
        for op in self.var:
            for w in self.var[op]:
                for s in self.var[op][w]:
                    # var = x * y * v(z的替代变量)
                    self.AddConstr(self.var[op][w][s] <= self.x[op][w])
                    self.AddConstr(self.var[op][w][s] <= self.y[w][s])
                    self.AddConstr(self.var[op][w][s] <= self.v[op][s])
                    self.AddConstr(self.x[op][w] + self.y[w][s] + self.v[op][s] - 2 <= self.var[op][w][s])

    def addFixedConstr(self):
        """
        固定（设备，工序，工人，工位）约束
        :return:
        """
        for m, _op, w, s in self.data_loader.fixed_alloc:
            op = m, _op
            self.AddConstr(self.x[op][w] == 1)
            self.AddConstr(self.y[w][s] == 1)
            self.AddConstr(self.v[op][s] == 1)
            self.AddConstr(self.var[op][w][s] == 1)

    def addCircleConstr(self):
        """
        工件圈数约束
        :return:
        """
        for parts in self.graph:
            for part in parts:
                list_gap = []
                list_ops = self.data_loader.partToOps[part]
                # 每个工件分配的工位数字
                listStNum = [[self.z[op][s] * self.data_loader.stToIdx[s] for s in self.z[op]] for op in list_ops]
                for i in range(len(listStNum) - 1):
                    preStNum = self.Sum(listStNum[i])
                    nextStNum = self.Sum(listStNum[i+1])
                    """
                    1.原始变量x，代表工序i+1到工序i对应工位的数字之差（若该数字之差小于0，即代表对应的工件存在绕圈）
                    2.x的取值范围是[-(StCnt-1), StCnt-1]，且是整数
                    3.若x属于[-(StCnt-1),-1]，则新变量z取1，代表绕了一圈；否则取0
                    """
                    x = nextStNum - preStNum
                    # 引入0-1辅助变量x1,x2
                    name1 = "circle_{0}_{1}_{2}_{3}_1".format(part, str(i),
                                                              tpl_to_str(list_ops[i]), tpl_to_str(list_ops[i+1]))
                    name2 = "circle_{0}_{1}_{2}_{3}_2".format(part, str(i),
                                                              tpl_to_str(list_ops[i]), tpl_to_str(list_ops[i+1]))
                    x1 = self.BoolVar(name1)
                    x2 = self.BoolVar(name2)
                    self.AddConstr(x1 + x2 == 1)
                    # 新的目标变量
                    z = 1 * x1 + 0 * x2
                    # 原始变量与辅助变量之间的约束关系
                    self.AddConstr(-1 * (self.StCnt-1) * x1 <= x)
                    self.AddConstr(x <= -1 * x1 + (self.StCnt-1) * x2)
                    # 添加新变量
                    list_gap.append(z)
                # 当前工件的绕圈数
                expr = self.Sum(list_gap)
                MaxCycleCnt = np.maximum(1, self.data_loader.conf["max_cycle_cnt"] - 1)
                self.AddConstr(expr <= MaxCycleCnt)

    def addRevisitedStConstr(self):
        """
        重复入站约束
        1.固定设备所在工位的重复入站数无限制，其他设备所在站位重复入站数上限不超过2次
        2.所有工位的重复入站的总数量小于max_revisited_station_count
        :return:
        """
        for s in self.data_loader.listStations:
            if s in self.data_loader.listFixSt:
                continue
            list_cnt = []
            for parts in self.graph:
                for part in parts:
                    # 统计(s,part,i)的入站情况
                    curPartInfo = []
                    list_ops = self.data_loader.partToOps[part]
                    N = len(list_ops)
                    # 原始变量x：当前部件part有多少道工序在当前工位s，取值大于1时，意味着重复入站
                    cur_st = [self.z[op][_s] for op in self.z if op in list_ops for _s in self.z[op] if _s == s]
                    cur_st.insert(0, 0)
                    for i in range(len(cur_st) - 1):
                        # 原始变量（入站数）
                        x = cur_st[i+1] - cur_st[i]
                        # 引入0-1辅助变量x1,x2，统计(s,part,i)的入站情况
                        name1 = "cnt_{0}_{1}_{2}_1".format(s, part, str(i))
                        name2 = "cnt_{0}_{1}_{2}_2".format(s, part, str(i))
                        x1 = self.BoolVar(name1)
                        x2 = self.BoolVar(name2)
                        self.AddConstr(x1 + x2 == 1)
                        # 新的变量y：当x取值大于1时，y=1，表示有新入站；否则表示没有新入站
                        y = 0 * x1 + 1 * x2
                        # 原始变量x与辅助变量x1,x2之间的约束关系
                        self.AddConstr(-1 * x1 + EPSILON * x2 <= x)
                        self.AddConstr(x <= x2)
                        # 累计单个part的重复入站次数
                        curPartInfo.append(y)
                    # 原始变量x的可能取值是(0,1,大于1)，(0,1)表示没有重复入站，其他表示有重复入站
                    x = self.Sum(curPartInfo)
                    # 引入0-1辅助变量x1,x2，统计(s,part)的重复入站情况
                    name1 = "revisited_{0}_{1}_1".format(s, part)
                    name2 = "revisited_{0}_{1}_2".format(s, part)
                    x1 = self.BoolVar(name1)
                    x2 = self.BoolVar(name2)
                    self.AddConstr(x1 + x2 == 1)
                    # 新的变量y：当x取值大于1时，y=1，表示重复入站；否则表示没有重复入站
                    y = 0 * x1 + 1 * x2
                    # 原始变量x与辅助变量x1,x2之间的约束关系
                    self.AddConstr(1 - x1 <= x)
                    self.AddConstr(x <= x2 * (N - 2) + x1)
                    # 累计s的重复入站数
                    list_cnt.append(y)
            # 约束：当前站位的重复入站数上限不超过2次
            expr = self.Sum(list_cnt)
            self.AddConstr(expr <= 2)

    def addObj1(self):
        """
        最小化最大节拍
        :return:
        """
        # 设立变量：最大节拍
        self.max_tt = self.NumVar(0, self.M, name="max_tt")
        for w in self.data_loader.listWorkers:
            # 每个员工的节拍
            list_t = [self.x[op][_w] * self.data_loader.wkTimeMap[(op[1], _w)] for op in self.x for _w in self.x[op]
                      if _w == w]
            t = self.Sum(list_t)
            # 添加约束
            self.AddConstr(t <= self.max_tt)
            # 记录每个员工的节拍
            self.tt[w] = t

    def addObj2(self):
        """
        最小化节拍波动率
        :return:
        """
        # 所有员工的平均节拍
        avg_t = self.Sum([t for t in self.tt.values()]) / self.WkCnt
        for w, t in self.tt.items():
            # 每个员工对应的大M
            OpCnt = len(self.data_loader.wkToAvailOps[w])
            M = self.MaxT * OpCnt
            # 原始变量
            x = t - avg_t
            # 引入新变量，等同于 abs(x)
            new_x = self.NumVar(0, M, name="obj_{0}".format(w))
            # 构建新变量和原始变量的约束关系
            self.AddConstr(x <= new_x)
            self.AddConstr(-1 * x <= new_x)
            self.AddConstr(new_x <= self.data_loader.conf["vol_rate"] * avg_t)
            # 累加每个员工的节拍波动
            self.obj.append(new_x)

    def minObj(self):
        obj = self.W1 * self.max_tt + self.W2 * self.Sum(self.obj)
        self.solver.Minimize(obj)

    def get_solution(self):
        data = [[p, w, s] for p in self.var for w in self.var[p] for s in self.var[p][w]
                if self.var[p][w][s].solution_value() > 0]
        df = pl.DataFrame(data=data, schema=["op", "worker_code", "station_code"], orient="row")
        df = df \
            .with_columns([
                pl.col("op").apply(lambda x: x[1], return_dtype=pl.Utf8).alias("operation")
            ]) \
            .with_columns([
                pl.col("operation").map_dict(self.opToIdx, return_dtype=pl.Int64).alias("operation_number"),
                pl.col("operation").map_dict(self.data_loader.opToPart, return_dtype=pl.Utf8).alias("part_code")
            ])
        df_st = self.data_loader.dataset.df_station \
            .rename({"st_code": "station_code"}) \
            .select(["station_code", "line_id"])
        df = df \
            .join(df_st, on=["station_code"], how="left") \
            .sort(by=["operation_number", "line_id"]) \
            .select(["line_id", "station_code", "worker_code", "operation", "operation_number", "part_code"])
        df.to_pandas().to_csv("df.csv", index=False)
        return df
