import heapq
from qiskit.transpiler.exceptions import TranspilerError

class DuostraRouter:
    """
    Implements the routing component of the Duostra qubit mapping algorithm.
    Responsible for finding the optimal path and SWAP sequences required to
    bring two logical qubits adjacent on the physical coupling map.
    """
    def __init__(self, coupling_map, timing_model):
        """
        Initializes the router with a hardware topology and execution timings.
        
        Args:
            coupling_map (CouplingMap): The physical hardware topology.
            timing_model (dict): Execution times for '1q', '2q', and 'swap' operations.
        """
        self.coupling_map = coupling_map
        self.timing_model = timing_model
        # Precompute an adjacency list (undirected neighbors) for faster pathfinding
        self.neighbors = {n: set() for n in range(self.coupling_map.size())}
        for edge in self.coupling_map.get_edges():
            self.neighbors[edge[0]].add(edge[1])
            self.neighbors[edge[1]].add(edge[0])

    def route(self, s0, s1, occupied_time):
        """
        Finds the optimal path between physical qubits s0 and s1 minimizing the final
        occupied time using a bilateral Dijkstra-style search.

        Args:
            s0 (int): First physical qubit index.
            s1 (int): Second physical qubit index.
            occupied_time (list[int]): Current temporal occupancy of each physical qubit.

        Returns:
            swaps (list of tuples): Ordered sequence of physical SWAPs (p_a, p_b).
            final_time (int): The resulting occupied time when these SWAPs complete.
        """
        # If qubits are already on the same node (edge case), no swaps needed
        if s1 == s0:
            return [], occupied_time[s0]
            
        # If qubits are already adjacent on the coupling map, they can execute a 2Q gate now
        if s1 in self.neighbors[s0]:
            return [], max(occupied_time[s0], occupied_time[s1])

        num_qubits = self.coupling_map.size()
        # Initialize node states for Dijkstra's algorithm.
        # 'source' tracks which start node (s0 or s1) reached this node first.
        nodes = {n: {'source': None, 'cost': float('inf'), 'visited': False, 'seen': False, 'parent': None} for n in range(num_qubits)}
        
        # Setup source s0
        nodes[s0]['source'] = s0
        nodes[s0]['cost'] = occupied_time[s0]
        nodes[s0]['seen'] = True
        nodes[s0]['visited'] = True
        
        # Setup source s1
        nodes[s1]['source'] = s1
        nodes[s1]['cost'] = occupied_time[s1]
        nodes[s1]['seen'] = True
        nodes[s1]['visited'] = True
        
        # Priority queue stores tuples of (current_cost, physical_qubit_index)
        pq = []
        
        def push_unseen_neighbors(m):
            """Helper to push unvisited neighbors of node 'm' into the priority queue."""
            for v in self.neighbors[m]:
                if not nodes[v]['seen'] and not nodes[v]['visited']:
                    nodes[v]['source'] = nodes[m]['source']
                    nodes[v]['seen'] = True
                    nodes[v]['parent'] = m # Track path for reconstruction
                    
                    # Cost is the max of the neighbor's occupied time and the current path's time, 
                    # plus the time taken to perform a SWAP gate
                    cost = max(occupied_time[v], nodes[m]['cost']) + self.timing_model['swap']
                    nodes[v]['cost'] = cost
                    heapq.heappush(pq, (cost, v))

        # Enqueue initial neighbors of both sources
        push_unseen_neighbors(s0)
        push_unseen_neighbors(s1)
        
        converge_m = None
        converge_v = None
        
        # Process the queue until paths from s0 and s1 intersect
        while pq:
            cost, m = heapq.heappop(pq)
            if nodes[m]['visited']:
                continue
                
            nodes[m]['visited'] = True
            
            # Check for convergence: if an adjacent node was reached from the OTHER source,
            # we have successfully found a complete path between s0 and s1.
            found_convergence = False
            for v in self.neighbors[m]:
                if nodes[v]['visited'] and nodes[v]['source'] != nodes[m]['source']:
                    converge_m = m    # Endpoint of path from one source
                    converge_v = v    # Endpoint of path from the other source
                    found_convergence = True
                    break
                    
            if found_convergence:
                break
                
            # Continue expanding
            push_unseen_neighbors(m)
            
        # Error out if the coupling graph is disconnected
        if converge_m is None or converge_v is None:
            raise TranspilerError("Coupling map is disconnected; cannot route.")

        # Reconstruct path from convergence point back to the respective sources
        path_m = []
        curr = converge_m
        while curr is not None:
            path_m.append(curr)
            curr = nodes[curr]['parent']
        path_m.reverse()   # Order from source to convergence point
        
        path_v = []
        curr = converge_v
        while curr is not None:
            path_v.append(curr)
            curr = nodes[curr]['parent']
        path_v.reverse()   # Order from source to convergence point
        
        # Orient the paths so full_path_s0 starts at s0 and full_path_s1 starts at s1
        if nodes[converge_m]['source'] == s0:
            full_path_s0 = path_m
            full_path_s1 = path_v
        else:
            full_path_s0 = path_v
            full_path_s1 = path_m

        swaps = []
        # The final execution time is bounded by whichever path section took longer
        final_time = max(nodes[converge_m]['cost'], nodes[converge_v]['cost'])
        
        # Convert path node sequences into explicit SWAP operations
        for i in range(len(full_path_s0) - 1):
            swaps.append((full_path_s0[i], full_path_s0[i+1]))
            
        for i in range(len(full_path_s1) - 1):
            swaps.append((full_path_s1[i], full_path_s1[i+1]))
            
        return swaps, final_time
