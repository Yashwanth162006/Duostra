from qiskit.transpiler.basepasses import TransformationPass
from qiskit.dagcircuit import DAGCircuit, DAGOpNode
from qiskit.circuit.library import SwapGate
from qiskit.transpiler.layout import Layout
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.circuit import AncillaRegister

from router import DuostraRouter
from scheduler import DuostraScheduler

class DuostraSwap(TransformationPass):
    """
    Qiskit TransformationPass that implements the Duostra qubit mapping algorithm.
    """
    def __init__(self, coupling_map, scheduler_mode="SP", depth=1, timing_model=None):
        super().__init__()
        self.coupling_map = coupling_map
        self.scheduler_mode = scheduler_mode
        self.depth = depth
        self.timing_model = timing_model or {'1q': 1, '2q': 2, 'swap': 6}
        self.router = DuostraRouter(coupling_map, self.timing_model)
        self.scheduler = DuostraScheduler(scheduler_mode, depth, self.timing_model, coupling_map)

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        if len(dag.qregs) != 1:
            raise TranspilerError("DuostraSwap requires a single qreg.")

        # Build initial mapped layout mapping
        initial_layout = getattr(self.property_set, 'layout', None)
        num_qubits = self.coupling_map.size()
        
        if initial_layout is None:
            initial_layout = Layout.generate_trivial_layout(*dag.qregs.values())
        
        # Ensure the layout map size == coupling_map.size()
        v2p = {v: p for v, p in initial_layout.get_virtual_bits().items()}
        
        # Maintain direct logic qubit object to physical index maps
        logical_to_physical = {q: p for q, p in v2p.items()}
        physical_to_logical = {p: q for q, p in v2p.items()}
        # For unused physical bits
        for i in range(num_qubits):
            if i not in physical_to_logical:
                physical_to_logical[i] = None

        occupied_time = [0] * num_qubits
        
        mapped_dag = dag.copy_empty_like()
        
        node_dependencies = {node: set() for node in dag.op_nodes()}
        in_degree = {node: 0 for node in dag.op_nodes()}
            
        for edge in dag.edges():
            if isinstance(edge[0], DAGOpNode) and isinstance(edge[1], DAGOpNode):
                if edge[1] not in node_dependencies[edge[0]]:
                    node_dependencies[edge[0]].add(edge[1])
                    in_degree[edge[1]] += 1
                    
        unmapped_nodes = [node for node in dag.op_nodes() if in_degree[node] == 0]
        
        def commit_node(node):
            if len(node.qargs) == 1:
                p = logical_to_physical[node.qargs[0]]
                occupied_time[p] += self.timing_model['1q']
            elif len(node.qargs) == 2:
                p0 = logical_to_physical[node.qargs[0]]
                p1 = logical_to_physical[node.qargs[1]]
                t_exec = max(occupied_time[p0], occupied_time[p1]) + self.timing_model['2q']
                occupied_time[p0] = t_exec
                occupied_time[p1] = t_exec
                
            mapped_dag.apply_operation_back(node.op, node.qargs, node.cargs)
            
            for succ in node_dependencies[node]:
                in_degree[succ] -= 1
                if in_degree[succ] == 0:
                    unmapped_nodes.append(succ)

        def apply_swap(p0, p1):
            l0 = physical_to_logical[p0]
            l1 = physical_to_logical[p1]
            
            logical_to_physical[l0] = p1
            physical_to_logical[p1] = l0
            
            if l1 is not None:
                logical_to_physical[l1] = p0
                physical_to_logical[p0] = l1
            else:
                physical_to_logical[p0] = None

            t_exec = max(occupied_time[p0], occupied_time[p1]) + self.timing_model['swap']
            occupied_time[p0] = t_exec
            occupied_time[p1] = t_exec
            
            if l1 is not None:
                mapped_dag.apply_operation_back(SwapGate(), [l0, l1], [])
            else:
                raise TranspilerError("Routing required an unallocated ancilla qubit, which is currently unhandled.")

        ancillas_added = False
        anc_reg = None
        for i in range(num_qubits):
            if physical_to_logical[i] is None:
                if not ancillas_added:
                    anc_reg = AncillaRegister(num_qubits, "ancilla")
                    mapped_dag.add_qreg(anc_reg)
                    ancillas_added = True
                physical_to_logical[i] = anc_reg[i]
                logical_to_physical[anc_reg[i]] = i

        while unmapped_nodes:
            gate_to_map = self.scheduler.select_next_gate(unmapped_nodes, occupied_time, self.router, logical_to_physical)
            if gate_to_map is None:
                # Should not happen unless waitlist is somehow unprocessable
                gate_to_map = unmapped_nodes[0]
            unmapped_nodes.remove(gate_to_map)
            
            if len(gate_to_map.qargs) == 2:
                p0 = logical_to_physical[gate_to_map.qargs[0]]
                p1 = logical_to_physical[gate_to_map.qargs[1]]
                swaps, _ = self.router.route(p0, p1, occupied_time)
                for swap in swaps:
                    apply_swap(swap[0], swap[1])
                    
            commit_node(gate_to_map)
            
        # Write back layout
        final_layout = Layout()
        for v, p in logical_to_physical.items():
            final_layout.add(v, p)
        self.property_set['final_layout'] = final_layout
        
        return mapped_dag
