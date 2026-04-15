class DuostraScheduler:
    """Schedules which operations to map next based on SP-Estimation or LE-Search."""
    def __init__(self, mode="SP", depth=1, timing_model=None, coupling_map=None):
        self.mode = mode
        self.depth = depth
        self.timing_model = timing_model or {'1q': 1, '2q': 2, 'swap': 6}
        self.coupling_map = coupling_map
        if self.coupling_map is not None:
            self.dist_matrix = self.coupling_map.distance_matrix

    def select_next_gate(self, waitlist, occupied_time, router, logical_to_physical):
        """
        waitlist: list of DAGOpNode
        logical_to_physical: dict mapping logic qubit objects to physical integers
        """
        if not waitlist:
            return None
        
        valid_2q = []
        single_q = []
        
        for node in waitlist:
            if len(node.qargs) != 2:
                single_q.append(node)
                # Single qubit gates and barriers don't violate coupling, we can immediately schedule them
            else:
                valid_2q.append(node)
                
        # If any single-qubit gates are ready, unconditionally schedule them next for throughput
        if single_q:
            return single_q[0]
            
        best_gate = None
        min_score = float('inf')
        
        if self.mode == "SP":
            for node in valid_2q:
                p0 = logical_to_physical[node.qargs[0]]
                p1 = logical_to_physical[node.qargs[1]]
                dist = self.dist_matrix[p0, p1]
                score = max(occupied_time[p0], occupied_time[p1]) + (dist - 1) * self.timing_model['swap']
                if score < min_score:
                    min_score = score
                    best_gate = node
                    
        elif self.mode == "LE":
            for node in valid_2q:
                p0 = logical_to_physical[node.qargs[0]]
                p1 = logical_to_physical[node.qargs[1]]
                _, final_time = router.route(p0, p1, occupied_time)
                score = final_time
                if score < min_score:
                    min_score = score
                    best_gate = node
        else:
            raise ValueError(f"Unknown mode {self.mode}")
            
        return best_gate
