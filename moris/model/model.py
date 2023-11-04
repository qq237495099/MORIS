from typing import List, Tuple, Dict
from ortools.linear_solver import pywraplp
import ortools.linear_solver.linear_solver_natural_api as lp

from moris.graph import Graph
from moris.data import DataLoader


SolverStatus = {0: 'OPTIMAL',
                1: 'FEASIBLE',
                2: 'INFEASIBLE',
                3: 'UNBOUNDED',
                4: 'ABNORMAL',
                6: 'NOT_SOLVED'}


def tpl_to_str(tpl: Tuple[str, str]):
    return ";".join(tpl)


class Model:
    def __init__(self, data_loader: DataLoader):
        self.x = {}
        self.y = {}
        self.z = {}
        self.w = {}
        self.solver = self.init_solver()
        self.data_loader = data_loader
        self.graph = Graph(data_loader.graph)

    @staticmethod
    def init_solver() -> pywraplp.Solver:
        return pywraplp.Solver.CreateSolver("SCIP")

    @property
    def StCnt(self) -> int:
        return self.data_loader.StCnt

    @property
    def WkCnt(self) -> int:
        return self.data_loader.WkCnt

    @property
    def PartCnt(self) -> int:
        return self.graph.NumParts

    @property
    def MaxT(self) -> float:
        return max(list(self.data_loader.wkTimeMap.values()))

    @property
    def OpCnt(self) -> int:
        return len(self.data_loader.opToPart)

    @property
    def M(self) -> float:
        return self.MaxT * self.OpCnt

    @property
    def VolRate(self) -> float:
        return self.data_loader.conf["vol_rate"]

    @property
    def W1(self) -> float:
        return self.data_loader.conf["upph_w"]

    @property
    def W2(self) -> float:
        return self.data_loader.conf["vol_w"]

    @property
    def Inf(self):
        return self.solver.infinity()

    @property
    def Vars(self) -> List[pywraplp.Variable]:
        return self.solver.variables()

    def LookupVariable(self, name: str):
        return self.solver.LookupVariable(name)

    def VarVal(self, var: str):
        return self.LookupVariable(var).solution_value()

    def BoolVar(self, name: str):
        return self.solver.BoolVar(name)

    def NumVar(self, lb: float, ub: float, name: str):
        return self.solver.NumVar(lb, ub, name)

    def IntVar(self, lb: int, ub: int, name: str):
        return self.solver.IntVar(lb, ub, name)

    def Sum(self, var: List[lp.VariableExpr]):
        return self.solver.Sum(var)

    def create_x(self, x: Tuple[str, str], y: str):
        # 工序x分配给工人y
        name = "x_{0}_{1}".format(tpl_to_str(x), y)
        var = self.BoolVar(name=name)
        self.x.setdefault(x, {})
        self.x[x][y] = var

    def create_y(self, x: str, y: str):
        # 工人x分配到工位y
        name = "y_{0}_{1}".format(x, y)
        var = self.BoolVar(name=name)
        self.y.setdefault(x, {})
        self.y[x][y] = var

    def create_z(self, x: Tuple[str, str], y: str):
        # 工序x通过主体工人到分配工位y
        name = "z_{0}_{1}".format(tpl_to_str(x), y)
        var = self.BoolVar(name=name)
        self.z.setdefault(x, {})
        self.z[x][y] = var

    def create_w(self, x: str, y: str):
        # 工位x分配设备y
        name = "w_{0}_{1}".format(x, y)
        var = self.BoolVar(name=name)
        self.w.setdefault(x, {})
        self.w[x][y] = var

    @property
    def opToIdx(self) -> Dict[str, int]:
        dictOpToIdx = {}
        s_cnt = 1
        for parts in self.graph:
            for part in parts:
                list_ops = self.data_loader.partToOps[part]
                opToIdx = {op[1]: idx + s_cnt for idx, op in enumerate(list_ops)}
                dictOpToIdx.update(opToIdx)
                s_cnt += len(list_ops)
        return dictOpToIdx

    def AddConstr(self, constr: lp.LinearConstraint):
        self.solver.Add(constr)

    @property
    def ObjValue(self):
        return self.solver.Objective().Value()

    def solveModel(self):
        status = self.solver.Solve()
        print(status)
        if status in [pywraplp.Solver.FEASIBLE, pywraplp.Solver.OPTIMAL]:
            print('Solution:')
            print('Objective value =', self.ObjValue)
            print('Problem solved in %f milliseconds' % self.solver.wall_time())
        else:
            print("The problem does not have solution")
