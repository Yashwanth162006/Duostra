from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap, PassManager
from qiskit.transpiler.passes import TrivialLayout
from qiskit.circuit.random import random_circuit

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'UGP'))

from duostra_qiskit import DuostraSwap

# Create a complex coupling map based on a 3x3 grid (9 qubits)
print("Creating a 3x3 grid topology...")
coupling_map = CouplingMap.from_grid(3, 3)

# Generate a simple 10-gate quantum circuit
print("Generating a simple 10-gate quantum circuit...")
qc = QuantumCircuit(9)

# 10 simple gates (creating entanglement across the grid)
qc.h(0)
qc.h(2)
qc.h(6)
qc.h(8)

# The following CX gates will require swaps because they span across the 3x3 grid
qc.cx(0, 8)  # Top-left to Bottom-right
qc.cx(2, 6)  # Top-right to Bottom-left
qc.cx(1, 7)  # Top-middle to Bottom-middle
qc.cx(3, 5)  # Left-middle to Right-middle

# Two more gates to reach 10 gates exactly before measurement
qc.rx(3.14/2, 4)
qc.ry(3.14/2, 4)

qc.measure_all()

print("\nOriginal Complex Circuit ops:")
print(qc.count_ops())

try:
    qc.draw(output='mpl', filename='simple_10gate_original.png')
    print("Saved simple original circuit image as 'simple_10gate_original.png'")
except Exception as e:
    print("Could not draw original circuit:", e)

# Pass Manager setup
try:
    # 1. Test Duostra mapped with SP Scheduler
    pm_sp = PassManager()
    pm_sp.append(TrivialLayout(coupling_map))
    pm_sp.append(DuostraSwap(coupling_map, scheduler_mode="SP"))
    
    mapped_sp = pm_sp.run(qc)
    print("\nMapped Circuit ops with Shortest Path (SP) scheduler:")
    print(mapped_sp.count_ops())
    
    try:
        mapped_sp.draw(output='mpl', filename='simple_10gate_mapped_sp.png')
        print("Saved SP mapped circuit image as 'simple_10gate_mapped_sp.png'")
    except Exception as e:
        print("Could not draw SP mapped circuit:", e)

    # 2. Test Duostra mapped with LE Scheduler
    pm_le = PassManager()
    pm_le.append(TrivialLayout(coupling_map))
    pm_le.append(DuostraSwap(coupling_map, scheduler_mode="LE"))
    
    mapped_le = pm_le.run(qc)
    print("\nMapped Circuit ops with Limitedly-Exhaustive (LE) scheduler:")
    print(mapped_le.count_ops())
    
    try:
        mapped_le.draw(output='mpl', filename='simple_10gate_mapped_le.png', idle_wires=False)
        print("Saved LE mapped circuit image as 'simple_10gate_mapped_le.png'")
    except Exception as e:
        print("Could not draw LE mapped circuit:", e)

except Exception as e:
    import traceback
    traceback.print_exc()
