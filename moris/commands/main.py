# https://pypi.tuna.tsinghua.edu.cn/simple
from moris.utils import *
from moris.model import OptModel
from moris.data import Dataset, DataLoader


@time_it(module="main")
def main():
    filename = "instance-1.txt"
    dataset = Dataset(filename)
    data_loader = DataLoader(dataset)
    # 初始化模型
    model = OptModel(data_loader)
    # 构建变量
    model.allocStToMach()
    model.allocOpToWks()
    model.allocWkToSts()
    model.allocOpToSts()
    model.create_var()
    # 添加约束
    model.addVarConstr()
    model.addFixedConstr()
    model.addCircleConstr()
    model.addRevisitedStConstr()
    # 设置目标
    model.addObj1()
    model.addObj2()
    model.minObj()
    # 求解
    model.solveModel()
    print(model.ObjValue)
    df = model.get_solution()
    print(df)
    print(model.OpCnt)


if __name__ == '__main__':
    main()
