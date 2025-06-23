"""
Quantum Random Number Generator Backend
Integrates Quantum-Simulator-Lite-at-Home for enhanced quantum simulation capabilities
"""

import numpy as np
from typing import List, Dict, Optional, Union
import json
import time
import hashlib

# Import the quantum simulator components
try:
    from quantum_simulator.core.gates import QuantumGate, GateType
    from quantum_simulator.core.simulator import QuantumSimulator
    from quantum_simulator.multi_framework import MultiFrameworkSimulator
    SIMULATOR_AVAILABLE = True
except ImportError:
    print("Warning: Quantum-Simulator-Lite-at-Home not found. Using Qiskit fallback.")
    SIMULATOR_AVAILABLE = False

# Fallback to Qiskit if the custom simulator is not available
if not SIMULATOR_AVAILABLE:
    try:
        from qiskit import QuantumCircuit, transpile
        from qiskit_aer import AerSimulator
        QISKIT_AVAILABLE = True
    except ImportError:
        QISKIT_AVAILABLE = False
        print("Error: Neither custom simulator nor Qiskit available.")


class QuantumRNGBackend:
    """
    Quantum Random Number Generator Backend
    Uses quantum superposition and measurement to generate truly random numbers
    """
    
    def __init__(self, shots: int = 1000, use_custom_simulator: bool = True):
        """
        Initialize the quantum RNG backend
        
        Args:
            shots: Number of shots to run for each random number generation
            use_custom_simulator: Whether to use the custom quantum simulator
        """
        self.shots = shots
        self.use_custom_simulator = use_custom_simulator and SIMULATOR_AVAILABLE
        
        if self.use_custom_simulator:
            self.simulator = MultiFrameworkSimulator(shots=shots)
        elif QISKIT_AVAILABLE:
            self.backend = AerSimulator()
        else:
            raise RuntimeError("No quantum simulator available")
    
    def generate_single_random_bit(self) -> int:
        """
        Generate a single random bit using quantum superposition
        
        Returns:
            int: 0 or 1
        """
        if self.use_custom_simulator:
            return self._generate_with_custom_simulator(1)[0]
        else:
            return self._generate_with_qiskit(1)[0]
    
    def generate_random_bits(self, num_bits: int) -> List[int]:
        """
        Generate multiple random bits
        
        Args:
            num_bits: Number of random bits to generate
            
        Returns:
            List[int]: List of random bits (0s and 1s)
        """
        if self.use_custom_simulator:
            return self._generate_with_custom_simulator(num_bits)
        else:
            return self._generate_with_qiskit(num_bits)
    
    def generate_random_number(self, min_val: int = 0, max_val: int = 255) -> int:
        """
        Generate a random number within a specified range
        
        Args:
            min_val: Minimum value (inclusive)
            max_val: Maximum value (inclusive)
            
        Returns:
            int: Random number in the specified range
        """
        range_size = max_val - min_val + 1
        num_bits_needed = (range_size - 1).bit_length()
        
        # Generate enough bits to cover the range
        bits = self.generate_random_bits(num_bits_needed)
        
        # Convert bits to number
        number = sum(bit << i for i, bit in enumerate(bits))
        
        # Ensure the number is within range
        while number >= range_size:
            # Generate additional bits if needed
            additional_bits = self.generate_random_bits(num_bits_needed)
            number = sum(bit << i for i, bit in enumerate(additional_bits))
        
        return min_val + number
    
    def generate_random_bytes(self, num_bytes: int) -> bytes:
        """
        Generate random bytes
        
        Args:
            num_bytes: Number of bytes to generate
            
        Returns:
            bytes: Random bytes
        """
        bits = self.generate_random_bits(num_bytes * 8)
        
        # Convert bits to bytes
        byte_array = bytearray()
        for i in range(0, len(bits), 8):
            byte = 0
            for j in range(8):
                if i + j < len(bits):
                    byte |= bits[i + j] << j
            byte_array.append(byte)
        
        return bytes(byte_array)
    
    def _generate_with_custom_simulator(self, num_bits: int) -> List[int]:
        """
        Generate random bits using the custom quantum simulator
        """
        # Create a simple circuit with Hadamard gates for superposition
        circuit = []
        for i in range(num_bits):
            circuit.append(QuantumGate(GateType.H, [i]))
            circuit.append(QuantumGate(GateType.MEASURE, [i], classical_bits=[i]))
        
        # Run the simulation
        results = self.simulator.simulate(circuit, framework='native')
        
        # Extract the measurement results
        if 'counts' in results:
            # Get the most frequent result
            most_frequent = max(results['counts'].items(), key=lambda x: x[1])[0]
            return [int(bit) for bit in most_frequent]
        else:
            # Fallback: generate based on probabilities
            return [np.random.choice([0, 1]) for _ in range(num_bits)]
    
    def _generate_with_qiskit(self, num_bits: int) -> List[int]:
        """
        Generate random bits using Qiskit (fallback)
        """
        qc = QuantumCircuit(num_bits, num_bits)
        
        # Apply Hadamard gate to each qubit to create superposition
        for i in range(num_bits):
            qc.h(i)
            qc.measure(i, i)
        
        # Compile and run
        compiled_circuit = transpile(qc, self.backend)
        job = self.backend.run(compiled_circuit, shots=self.shots)
        result = job.result()
        counts = result.get_counts(compiled_circuit)
        
        # Get the most frequent result
        most_frequent = max(counts.items(), key=lambda x: x[1])[0]
        return [int(bit) for bit in most_frequent]
    
    def generate_entropy_pool(self, pool_size: int = 1024) -> Dict[str, Union[str, int, float]]:
        """
        Generate an entropy pool for cryptographic applications
        
        Args:
            pool_size: Size of the entropy pool in bits
            
        Returns:
            Dict containing entropy pool data and metadata
        """
        start_time = time.time()
        
        # Generate random bits
        bits = self.generate_random_bits(pool_size)
        
        # Calculate entropy metrics
        ones_count = sum(bits)
        zeros_count = pool_size - ones_count
        
        # Calculate Shannon entropy
        p_ones = ones_count / pool_size
        p_zeros = zeros_count / pool_size
        
        if p_ones > 0 and p_zeros > 0:
            shannon_entropy = -(p_ones * np.log2(p_ones) + p_zeros * np.log2(p_zeros))
        else:
            shannon_entropy = 0.0
        
        # Create entropy pool
        entropy_pool = {
            'bits': ''.join(map(str, bits)),
            'hex': ''.join(f'{sum(bits[i:i+4]) * (2**(3-j)) for j in range(4)}' 
                         for i in range(0, len(bits), 4)),
            'ones_count': ones_count,
            'zeros_count': zeros_count,
            'balance_ratio': ones_count / pool_size,
            'shannon_entropy': shannon_entropy,
            'generation_time': time.time() - start_time,
            'pool_size': pool_size,
            'timestamp': time.time()
        }
        
        return entropy_pool
    
    def generate_cryptographic_key(self, key_length: int = 256) -> str:
        """
        Generate a cryptographic key using quantum randomness
        
        Args:
            key_length: Length of the key in bits
            
        Returns:
            str: Hexadecimal representation of the key
        """
        random_bytes = self.generate_random_bytes(key_length // 8)
        return random_bytes.hex()
    
    def test_randomness(self, num_samples: int = 1000) -> Dict[str, float]:
        """
        Test the quality of randomness using statistical tests
        
        Args:
            num_samples: Number of samples to test
            
        Returns:
            Dict containing statistical test results
        """
        # Generate samples
        samples = [self.generate_single_random_bit() for _ in range(num_samples)]
        
        # Calculate statistics
        ones_count = sum(samples)
        zeros_count = num_samples - ones_count
        
        # Balance test
        balance = abs(ones_count - zeros_count) / num_samples
        
        # Runs test (simplified)
        runs = 1
        for i in range(1, len(samples)):
            if samples[i] != samples[i-1]:
                runs += 1
        
        expected_runs = (2 * ones_count * zeros_count) / num_samples + 1
        runs_ratio = runs / expected_runs if expected_runs > 0 else 1.0
        
        return {
            'balance': balance,
            'runs_ratio': runs_ratio,
            'ones_ratio': ones_count / num_samples,
            'zeros_ratio': zeros_count / num_samples,
            'total_runs': runs,
            'expected_runs': expected_runs
        }


# Example usage and testing
if __name__ == "__main__":
    # Initialize the quantum RNG backend
    qrng = QuantumRNGBackend(shots=1000, use_custom_simulator=True)
    
    print("Quantum Random Number Generator Backend")
    print("=" * 40)
    
    # Generate some random bits
    print(f"Single random bit: {qrng.generate_single_random_bit()}")
    print(f"10 random bits: {qrng.generate_random_bits(10)}")
    
    # Generate random numbers
    print(f"Random number (0-255): {qrng.generate_random_number(0, 255)}")
    print(f"Random number (1-100): {qrng.generate_random_number(1, 100)}")
    
    # Generate cryptographic key
    key = qrng.generate_cryptographic_key(256)
    print(f"256-bit cryptographic key: {key}")
    
    # Test randomness
    test_results = qrng.test_randomness(1000)
    print(f"Randomness test results: {test_results}")
    
    # Generate entropy pool
    entropy_pool = qrng.generate_entropy_pool(1024)
    print(f"Entropy pool generated in {entropy_pool['generation_time']:.3f} seconds")
    print(f"Shannon entropy: {entropy_pool['shannon_entropy']:.3f}") 