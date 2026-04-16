class DuostraScheduler:
    """
    Schedules which quantum operation from the available waitlist should be 
    mapped next. It supports choosing via Shortest-Path (SP) Estimation or 
    Limitedly-Exhaustive (LE) Search.
    """
    def __init__(self, mode="SP", depth=1, timing_model=None, coupling_map=None):
        """
        Args:
            mode (str): Scheduling heuristic to use ("SP" or "LE").
            depth (int): Optional depth parameter for lookahead (default is 1).
            timing_model (dict): Gate execution delays to track availability.
            coupling_map (CouplingMap): Physical graph needed for SP distance queries.
        """
        self.mode = mode
        self.depth = depth
        self.timing_model = timing_model or {'1q': 1, '2q': 2, 'swap': 6}
        self.coupling_map = coupling_map
        # Precompute distance matrix for SP mode if a coupling map is provided
        if self.coupling_map is not None:
            self.dist_matrix = self.coupling_map.distance_matrix

    def select_next_gate(self, waitlist, occupied_time, router, logical_to_physical):
        """
        Selects the most optimal gate from the waitlist to map next.

        Args:
            waitlist (list[DAGOpNode]): Executable operations whose dependencies are met.
            occupied_time (list[int]): Temporal availability of each physical qubit.
            router (DuostraRouter): Router instance used for LE cost estimation.
            logical_to_physical (dict): Current logical-to-physical qubit assignments.

        Returns:
            DAGOpNode: The optimal gate to process next, or None if waitlist is empty.
        """
        if not waitlist:
            return None
        
        valid_2q = []
        single_q = []
        
        # Separate operations into single-qubit and multi-qubit categories
        for node in waitlist:
            if len(node.qargs) != 2:
                # Single-qubit gates and barriers only affect one qubit at a time
                single_q.append(node)
            else:
                valid_2q.append(node)
                
        # If any single-qubit gates are ready, unconditionally schedule them next
        # to maximize throughput since they don't incur routing (SWAP) overhead.
        if single_q:
            return single_q[0]
            
        best_gate = None
        min_score = float('inf')
        
        # SP Mode: Estimate routing cost using static distance matrix
        if self.mode == "SP":
            for node in valid_2q:
                p0 = logical_to_physical[node.qargs[0]]
                p1 = logical_to_physical[node.qargs[1]]
                dist = self.dist_matrix[p0, p1]
                
                # Score is based on current availability plus estimated SWAP delay
                score = max(occupied_time[p0], occupied_time[p1]) + (dist - 1) * self.timing_model['swap']
                if score < min_score:
                    min_score = score
                    best_gate = node
                    
        # LE Mode: Perform actual mock-routing for each gate to find the true lowest cost
        elif self.mode == "LE":
            for node in valid_2q:
                p0 = logical_to_physical[node.qargs[0]]
                p1 = logical_to_physical[node.qargs[1]]
                # Use the router to find the exact finalized time if this gate were mapped next
                _, final_time = router.route(p0, p1, occupied_time)
                
                score = final_time
                if score < min_score:
                    min_score = score
                    best_gate = node
        else:
            raise ValueError(f"Unknown mode {self.mode}")
            
        return best_gate
