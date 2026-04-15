from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap, PassManager
from swap import DuostraSwap

# Create a small graph (a line of 4 qubits)
coupling_map = CouplingMap.from_line(4)

# Create a circuit requiring swaps
qc = QuantumCircuit(4)
qc.h(0)
qc.cx(0, 3) # not adjacent, distance 3
qc.cx(1, 2)
qc.measure_all()

print("Original Circuit:")
print(str(qc.draw(output='text', fold=-1)).encode('ascii', 'replace').decode('ascii'))
print("Original Circuit ops:", dict(qc.count_ops()))

try:
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit.transpiler.passes import TrivialLayout
    
    # --- SP Scheduler Test ---
    pass_manager_sp = PassManager()
    pass_manager_sp.append(TrivialLayout(coupling_map))
    pass_manager_sp.append(DuostraSwap(coupling_map, scheduler_mode="SP"))
    
    mapped_circ_sp = pass_manager_sp.run(qc)
    print("\n" + "="*50)
    print("Mapped Circuit with SP scheduler:")
    print(str(mapped_circ_sp.draw(output='text', fold=-1)).encode('ascii', 'replace').decode('ascii'))
    print("Mapped Circuit ops:", dict(mapped_circ_sp.count_ops()))

    # --- LE Scheduler Test ---
    pass_manager_le = PassManager()
    pass_manager_le.append(TrivialLayout(coupling_map))
    pass_manager_le.append(DuostraSwap(coupling_map, scheduler_mode="LE"))
    
    mapped_circ_le = pass_manager_le.run(qc)
    print("\n" + "="*50)
    print("Mapped Circuit with LE scheduler:")
    print(str(mapped_circ_le.draw(output='text', fold=-1)).encode('ascii', 'replace').decode('ascii'))
    print("Mapped Circuit ops:", dict(mapped_circ_le.count_ops()))
    
except Exception as e:
    import traceback
    traceback.print_exc()