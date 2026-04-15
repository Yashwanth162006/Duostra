import heapq
from qiskit.transpiler.exceptions import TranspilerError

class DuostraRouter:
    """This Function Implements the Duostra Algorithm"""
    def __init__(self, coupling_map, timing_model):
        self.coupling_map = coupling_map
        self.timing_model = timing_model
        # Precompute undirected neighbors for faster lookup
        self.neighbors = {n: set() for n in range(self.coupling_map.size())}
        for edge in self.coupling_map.get_edges():
            self.neighbors[edge[0]].add(edge[1])
            self.neighbors[edge[1]].add(edge[0])

    def route(self, s0, s1, occupied_time):
        """
        Finds the optimal path between s0 and s1 minimizing occupied time.
        Returns:
            swaps: list of tuples (p_a, p_b) of physical SWAPs to perform
            final_time: int, the occupied time after these SWAPs finish.
        """
        if s1 == s0:
            return [], occupied_time[s0]
            
        if s1 in self.neighbors[s0]:
            return [], max(occupied_time[s0], occupied_time[s1])

        num_qubits = self.coupling_map.size()
        nodes = {n: {'source': None, 'cost': float('inf'), 'visited': False, 'seen': False, 'parent': None} for n in range(num_qubits)}
        
        nodes[s0]['source'] = s0
        nodes[s0]['cost'] = occupied_time[s0]
        nodes[s0]['seen'] = True
        nodes[s0]['visited'] = True
        
        nodes[s1]['source'] = s1
        nodes[s1]['cost'] = occupied_time[s1]
        nodes[s1]['seen'] = True
        nodes[s1]['visited'] = True
        
        pq = []
        
        def push_unseen_neighbors(m):
            for v in self.neighbors[m]:
                if not nodes[v]['seen'] and not nodes[v]['visited']:
                    nodes[v]['source'] = nodes[m]['source']
                    nodes[v]['seen'] = True
                    nodes[v]['parent'] = m
                    
                    cost = max(occupied_time[v], nodes[m]['cost']) + self.timing_model['swap']
                    nodes[v]['cost'] = cost
                    heapq.heappush(pq, (cost, v))

        push_unseen_neighbors(s0)
        push_unseen_neighbors(s1)
        
        converge_m = None
        converge_v = None
        
        while pq:
            cost, m = heapq.heappop(pq)
            if nodes[m]['visited']:
                continue
                
            nodes[m]['visited'] = True
            
            found_convergence = False
            for v in self.neighbors[m]:
                if nodes[v]['visited'] and nodes[v]['source'] != nodes[m]['source']:
                    converge_m = m
                    converge_v = v
                    found_convergence = True
                    break
                    
            if found_convergence:
                break
                
            push_unseen_neighbors(m)
            
        if converge_m is None or converge_v is None:
            raise TranspilerError("Coupling map is disconnected; cannot route.")

        path_m = []
        curr = converge_m
        while curr is not None:
            path_m.append(curr)
            curr = nodes[curr]['parent']
        path_m.reverse()
        
        path_v = []
        curr = converge_v
        while curr is not None:
            path_v.append(curr)
            curr = nodes[curr]['parent']
        path_v.reverse()
        
        if nodes[converge_m]['source'] == s0:
            full_path_s0 = path_m
            full_path_s1 = path_v
        else:
            full_path_s0 = path_v
            full_path_s1 = path_m

        swaps = []
        final_time = max(nodes[converge_m]['cost'], nodes[converge_v]['cost'])
        
        for i in range(len(full_path_s0) - 1):
            swaps.append((full_path_s0[i], full_path_s0[i+1]))
            
        for i in range(len(full_path_s1) - 1):
            swaps.append((full_path_s1[i], full_path_s1[i+1]))
            
        return swaps, final_time
