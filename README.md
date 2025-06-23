# Quantum Random Number Generator

A demonstration of quantum computing principles applied to cryptography through the implementation of a truly random number generator using quantum mechanics.

## 🌟 Overview

This project demonstrates how quantum computers can be used in cryptography by implementing a fundamental entity: **Random Number Generators (RNGs)**. Unlike classical pseudorandom number generators, this implementation leverages the inherent randomness of quantum mechanics to generate truly random numbers.

## 🔬 Quantum Mechanics Concepts

### Key Principles Used

1. **Qubits**: Quantum bits that can exist in superposition of states
2. **Quantum Superposition**: The ability of a qubit to be in multiple states simultaneously
3. **No Cloning Theorem**: It's impossible to create an identical copy of an arbitrary unknown quantum state
4. **Quantum Measurement**: The process of observing a quantum state, which collapses it to a definite state

### Mathematical Foundation

A qubit is represented as a 2-dimensional complex vector:

|ψ⟩ = α|0⟩ + β|1⟩

Where α and β are complex numbers satisfying the normalization condition:
|α|² + |β|² = 1

## 🎯 How It Works

### Classical vs Quantum RNG

**Classical RNGs (PRNGs):**
- Deterministic algorithms that appear random
- Require a seed value
- Can be predicted if the seed is known
- Used in most applications today

**Quantum RNGs:**
- Leverage inherent quantum randomness
- No seed required
- Truly unpredictable
- Based on fundamental quantum mechanical principles

### Implementation Details

The quantum random number generator uses:

1. **Hadamard Gate (H)**: Creates superposition from computational basis states
   ```
   H = 1/√2 [1  1]
           [1 -1]
   ```

2. **Quantum Circuit**: 
   - Initialize qubit in |0⟩ state
   - Apply Hadamard gate to create superposition
   - Measure the qubit to get random 0 or 1

3. **Measurement**: The superposition collapses to either |0⟩ or |1⟩ with equal probability (50/50)





1. **Theoretical Background**: Explanation of quantum mechanics concepts
2. **Circuit Implementation**: Creating the quantum circuit with Qiskit
3. **Simulation**: Running the circuit on a quantum simulator
4. **Visualization**: Plotting results and probability distributions
5. **Analysis**: Statistical analysis of randomness

## 🔍 Example Output

When you run the quantum circuit, you'll see:

```
Measurement results: {'1': 514, '0': 510}

Running 1000 shots to demonstrate randomness:
Results after 1000 shots: {'0': 515, '1': 485}

Statistics:
Total shots: 1000
|0⟩ probability: 0.515
|1⟩ probability: 0.485
```

The results show approximately 50/50 distribution, demonstrating true randomness.

## 🛡️ Security Implications

### Why Quantum RNG is Superior

1. **True Randomness**: Based on fundamental quantum mechanical principles
2. **Unpredictability**: No amount of computational power can predict the outcome
3. **No Seed Dependency**: Unlike classical PRNGs, no seed is required
4. **Cryptographic Security**: Ideal for cryptographic applications

### Applications

- **Cryptography**: Key generation, nonces, random padding
- **Gambling**: Fair random number generation
- **Scientific Simulations**: Monte Carlo methods
- **Gaming**: Random events in games
- **Lotteries**: Fair random selection

## 🔬 Technical Details

### Quantum Circuit

The implemented circuit consists of:
1. One qubit initialized in |0⟩ state
2. Hadamard gate applied to create superposition
3. Measurement operation to extract random bit

### Probability Analysis

The Hadamard gate transforms the computational basis:
- H|0⟩ = |+⟩ = (|0⟩ + |1⟩)/√2
- H|1⟩ = |-⟩ = (|0⟩ - |1⟩)/√2

When measured in the computational basis:
- P(|0⟩) = |⟨0|ψ⟩|² = |α|²
- P(|1⟩) = |⟨1|ψ⟩|² = |β|²

For the |+⟩ state, both probabilities equal 1/2, giving true 50/50 randomness.

## 🌐 Real Quantum Hardware

While this demonstration uses Qiskit's Aer simulator, you can run the same circuit on real quantum hardware:

1. **IBM Quantum**: Access real quantum computers through IBM Quantum Experience
2. **Other Providers**: Various cloud quantum computing platforms
3. **Local Simulators**: For development and testing

## 📚 Further Reading

- [Qiskit Documentation](https://qiskit.org/documentation/)
- [Quantum Computing Concepts](https://qiskit.org/textbook/)
- [No Cloning Theorem](https://en.wikipedia.org/wiki/No-cloning_theorem)
- [Quantum Cryptography](https://en.wikipedia.org/wiki/Quantum_cryptography)

## 👨‍💻 Author

**Fazli Berk Ordek**
- GitHub: [@fazliberkordek](https://github.com/fazliberkordek)

---