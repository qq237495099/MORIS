import igraph as ig
from typing import List, Dict, Iterator


class Graph:
    __slots__ = ["g", "layerMap", "MaxLayer"]

    def __init__(self, g: ig.Graph):
        self.g = g
        self.layerMap = self.initLayerMap()
        self.updateVertexLayer()
        self.MaxLayer = max(list(self.layerMap.values()))

    def initLayerMap(self) -> Dict[str, int]:
        layerMap = {}
        root = self.DstVertex[0]
        layerMap[root] = 1
        cnt = 2
        completed = False
        curr_layers = [root]
        while not completed:
            # traversal current vertex layers
            for v in curr_layers:
                predecessors = self.predecessors(v)
                predecessors = [p for p in predecessors if p not in self.SrcVertex]
                if len(predecessors) == 0:
                    completed = True
                    break
                for p in predecessors:
                    layerMap[p] = cnt
                cnt += 1
                curr_layers = predecessors
        for p in self.SrcVertex:
            layerMap[p] = cnt
        return layerMap

    def updateVertexLayer(self):
        for v, idx in self.layerMap.items():
            self.update_vertex_attr(v, "layer", idx)

    def getVertexByLayer(self, layer: int) -> List[str]:
        return [v["name"] for v in self.g.vs if v["layer"] == layer]

    @property
    def NumParts(self) -> int:
        return len(self.Vertices)

    def __iter__(self) -> Iterator[List[str]]:
        for i in range(self.MaxLayer, 0, -1):
            vs = self.getVertexByLayer(i)
            yield vs

    def __contains__(self, v: str) -> bool:
        return v in self.Vertices

    def __getitem__(self, v: str) -> ig.Vertex:
        return self.vertex(v)

    def __str__(self):
        return "Graph with {0} Vertices and {1} Edges".format(str(len(self.Vertices)), str(len(self.Edges)))

    def __repr__(self):
        return self.__str__()

    @property
    def VertexToId(self):
        return {v["name"]: v.index for v in self.g.vs}

    @property
    def IdToVertex(self):
        return {idx: name for name, idx in self.VertexToId.items()}

    @property
    def EdgeToId(self):
        return {e["name"]: e.index for e in self.g.es}

    @property
    def IdToEdge(self):
        return {idx: name for name, idx in self.EdgeToId.items()}

    @property
    def Vertices(self) -> List[str]:
        return [v["name"] for v in self.g.vs]

    @property
    def Edges(self) -> List[str]:
        return [e["name"] for e in self.g.es]

    @property
    def SrcVertex(self) -> List[str]:
        return [v["name"] for v in self.g.vs if v.indegree() == 0]

    @property
    def DstVertex(self) -> List[str]:
        return [v["name"] for v in self.g.vs if v.outdegree() == 0]

    def vertex(self, v: str) -> ig.Vertex:
        """
        get 'ig.Vertex' by vertex name
        :param v: input vertex name
        :return:
        """
        return self.g.vs.find(name=v)

    def edge(self, e: str) -> ig.Edge:
        """
        get 'ig.Edge' by edge name
        :param e: input edge name
        :return:
        """
        return self.g.es.find(name=e)

    def update_vertex_attr(self, v: str, attr: str, value):
        """
        update vertex attr by setting the value
        :param v: vertex name
        :param attr: vertex attribute name
        :param value: vertex attribute value to be set
        :return:
        """
        v_idx = self.VertexToId[v]
        self.g.vs[v_idx][attr] = value

    def update_edge_attr(self, e: str, attr: str, value):
        """
        update edge attr by setting the value
        :param e: edge name
        :param attr: edge attribute name
        :param value: edge attribute value to be set
        :return:
        """
        e_idx = self.EdgeToId[e]
        self.g.es[e_idx][attr] = value

    def predecessors(self, v: str) -> List[str]:
        """
        find predecessors of the input vertex v
        :param v: vertex name
        :return: list of predecessors name
        """
        vertex = self.vertex(v)
        res = [v["name"] for v in vertex.predecessors()]
        return res

    def successors(self, v: str) -> List[str]:
        """
        find successors of the input vertex v
        :param v: vertex name
        :return: list of successors name
        """
        vertex = self.vertex(v)
        res = [v["name"] for v in vertex.successors()]
        return res

    def del_vs(self, vs: List[int]):
        """
        delete vertex series
        :param vs:
        :return:
        """
        self.g.delete_vertices(vs)

    def del_es(self, es: List[int]):
        """
        delete edge series
        :param es:
        :return:
        """
        self.g.delete_edges(es)
