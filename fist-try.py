import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
import os
import pickle
import time
import json
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
import multiprocessing as mp
from dataclasses import dataclass, field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("simulation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("self_healing")

@dataclass
class SimulationConfig:
    """Configuration class for simulation parameters"""
    # System parameters
    n_atoms: int = 1000
    box_size: List[float] = field(default_factory=lambda: [10.0, 10.0, 10.0])
    time_step: float = 0.002  # ps
    n_steps: int = 10000
    temperature: float = 300  # K
    pressure: float = 1.0  # bar
    
    # Material parameters
    material_type: str = "polymer"
    chain_length: int = 20
    crosslink_density: float = 0.1
    
    # ML model parameters
    model_type: str = "transformer"  # transformer, gcn, or hybrid
    model_dim: int = 256
    n_heads: int = 8
    n_layers: int = 6
    dropout: float = 0.1
    lr: float = 1e-4
    batch_size: int = 16
    epochs: int = 50
    
    # Healing parameters
    healing_trigger_threshold: float = 0.3
    healing_efficiency: float = 0.85
    healing_radius: float = 0.5
    
    # Computational parameters
    use_gpu: bool = True
    n_cores: int = max(1, mp.cpu_count() - 1)
    random_seed: int = 42
    save_interval: int = 100
    
    # File paths
    output_dir: str = "simulation_results"
    model_save_path: str = "trained_models"
    
    def __post_init__(self):
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() and self.use_gpu else "cpu")
        
        # Create directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.model_save_path, exist_ok=True)
        
        # Set random seeds for reproducibility
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.random_seed)
    
    def save(self, filepath: str) -> None:
        """Save configuration to JSON file"""
        # Convert to dictionary, excluding device attribute
        config_dict = {k: v for k, v in self.__dict__.items() if k != 'device'}
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=4)
    
    @classmethod
    def load(cls, filepath: str) -> 'SimulationConfig':
        """Load configuration from JSON file"""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)


class ForceField:
    """Class to handle interatomic force calculations with different potentials"""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        # Parameters for different atom type interactions
        self.pair_params = {
            # (type1, type2): (sigma, epsilon)
            (0, 0): (0.3, 0.1),
            (0, 1): (0.3, 0.12),
            (1, 1): (0.29, 0.15),
            (1, 2): (0.28, 0.17),
            (2, 2): (0.27, 0.2),
            (3, 3): (0.26, 0.22),
            (3, 4): (0.25, 0.25),
            (4, 4): (0.24, 0.3),
        }
        # Bond parameters for different atom types
        self.bond_params = {
            # (type1, type2): (r0, k)
            (0, 0): (0.15, 500.0),
            (1, 1): (0.15, 500.0),
            (1, 2): (0.14, 600.0),
            (2, 2): (0.13, 700.0),
            (2, 3): (0.13, 700.0),
            (3, 3): (0.12, 800.0),
            (3, 4): (0.12, 800.0),
            (4, 4): (0.11, 900.0),
        }
        # Angle parameters
        self.angle_params = {
            # (type1, type2, type3): (theta0, k)
            (1, 1, 1): (120.0, 100.0),
            (1, 2, 1): (109.5, 120.0),
        }
        
    def get_pair_params(self, type1: int, type2: int) -> Tuple[float, float]:
        """Get LJ parameters for a pair of atom types"""
        key = tuple(sorted([type1, type2]))
        return self.pair_params.get(key, (0.3, 0.1))  # Default if not found
    
    def get_bond_params(self, type1: int, type2: int) -> Tuple[float, float]:
        """Get bond parameters for a pair of atom types"""
        key = tuple(sorted([type1, type2]))
        return self.bond_params.get(key, (0.15, 500.0))  # Default if not found
    
    def get_angle_params(self, type1: int, type2: int, type3: int) -> Tuple[float, float]:
        """Get angle parameters for three atom types"""
        # Sort the first and third atoms (central atom must stay in the middle)
        if type1 > type3:
            type1, type3 = type3, type1
        key = (type1, type2, type3)
        return self.angle_params.get(key, (120.0, 100.0))  # Default if not found
    
    def lennard_jones(self, r: np.ndarray, sigma: float, epsilon: float) -> Tuple[np.ndarray, float]:
        """Calculate Lennard-Jones force and energy"""
        sr6 = (sigma/r)**6
        force = 24 * epsilon * (2 * sr6**2 - sr6) / r**2
        energy = 4 * epsilon * (sr6**2 - sr6)
        return force, energy
    
    def harmonic_bond(self, r: float, r0: float, k: float) -> Tuple[float, float]:
        """Calculate harmonic bond force and energy"""
        force = k * (r - r0)
        energy = 0.5 * k * (r - r0)**2
        return force, energy
    
    def harmonic_angle(self, theta: float, theta0: float, k: float) -> Tuple[float, float]:
        """Calculate harmonic angle force and energy"""
        # Convert degrees to radians
        theta0_rad = np.deg2rad(theta0)
        force = k * (theta - theta0_rad)
        energy = 0.5 * k * (theta - theta0_rad)**2
        return force, energy
    
    def morse_potential(self, r: float, r0: float, D: float, alpha: float) -> Tuple[float, float]:
        """Calculate Morse potential force and energy for weaker bonds that can break"""
        exp_term = np.exp(-alpha * (r - r0))
        energy = D * (1 - exp_term)**2
        force = 2 * D * alpha * exp_term * (1 - exp_term)
        return force, energy


class MDSimulator:
    """Enhanced molecular dynamics simulator with advanced features"""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.force_field = ForceField(config)
        self.positions = np.zeros((config.n_atoms, 3))
        self.velocities = np.zeros((config.n_atoms, 3))
        self.forces = np.zeros((config.n_atoms, 3))
        self.atom_types = np.zeros(config.n_atoms, dtype=int)
        self.masses = np.ones(config.n_atoms)  # Default mass = 1.0
        self.bonds = []
        self.angles = []
        self.damage_sites = []
        self.energies = {
            'kinetic': 0.0,
            'potential': 0.0,
            'total': 0.0
        }
        self.stress_tensor = np.zeros((3, 3))
        
    def initialize_system(self, material_type: Optional[str] = None) -> None:
        """Initialize the system with different material types"""
        if material_type is None:
            material_type = self.config.material_type
            
        logger.info(f"Initializing {material_type} system with {self.config.n_atoms} atoms")
        
        # Random positions in the box
        box = self.config.box_size
        self.positions = np.random.rand(self.config.n_atoms, 3) * box
        
        # Maxwell-Boltzmann distribution for velocities
        sigma = np.sqrt(self.config.temperature * 0.00831 / 1.0)  # in nm/ps
        self.velocities = np.random.normal(0, sigma, (self.config.n_atoms, 3))
        
        # Remove center of mass motion
        total_momentum = np.sum(self.velocities, axis=0)
        self.velocities -= total_momentum / self.config.n_atoms
        
        # Initialize atom types and structures based on material type
        if material_type == "polymer":
            self._initialize_polymer()
        elif material_type == "capsule":
            self._initialize_capsule()
        elif material_type == "hydrogel":
            self._initialize_hydrogel()
        elif material_type == "nanocomposite":
            self._initialize_nanocomposite()
        else:
            raise ValueError(f"Unknown material type: {material_type}")
        
        # Initialize bonds and angles
        self._setup_molecular_topology()
        
        # Calculate initial forces
        self.calculate_forces()
        
        logger.info(f"System initialized with {len(self.bonds)} bonds and {len(self.angles)} angles")
    
    def _initialize_polymer(self) -> None:
        """Initialize a polymer material"""
        chain_length = self.config.chain_length
        n_chains = self.config.n_atoms // chain_length
        
        # Create linear polymer chains
        for c in range(n_chains):
            start_idx = c * chain_length
            end_idx = min((c + 1) * chain_length, self.config.n_atoms)
            
            # Chain atoms (Type 1)
            self.atom_types[start_idx:end_idx] = 1
            
            # Set masses for polymer atoms
            self.masses[start_idx:end_idx] = 1.0
            
            # Create chain structure with more realistic positions
            chain_start = np.random.rand(3) * (np.array(self.config.box_size) - 2.0) + 1.0
            direction = np.random.randn(3)
            direction = direction / np.linalg.norm(direction)
            
            for i in range(start_idx, end_idx):
                # Position each atom with realistic bond length
                if i == start_idx:
                    self.positions[i] = chain_start
                else:
                    # Add some randomness to create bends in the chain
                    if i % 5 == 0:  # Every 5 atoms, change direction slightly
                        rand_vector = np.random.randn(3) * 0.1
                        direction = direction + rand_vector
                        direction = direction / np.linalg.norm(direction)
                    
                    self.positions[i] = self.positions[i-1] + direction * 0.15
        
        # Add reversible crosslinking sites (Type 2)
        crosslink_density = self.config.crosslink_density
        n_crosslinks = int(self.config.n_atoms * crosslink_density)
        
        crosslink_sites = np.random.choice(
            np.where(self.atom_types == 1)[0], 
            size=n_crosslinks, 
            replace=False
        )
        self.atom_types[crosslink_sites] = 2
    
    def _initialize_capsule(self) -> None:
        """Initialize a self-healing capsule-based material"""
        # Matrix material (Type 1)
        self.atom_types[:] = 1
        self.masses[:] = 1.0
        
        # Create microcapsules
        n_capsules = 8
        capsule_radius = 1.0
        
        for i in range(n_capsules):
            # Random center within the box (avoiding edges)
            center = np.random.rand(3) * (np.array(self.config.box_size) - 2*capsule_radius) + capsule_radius
            
            # Find atoms within the capsule radius
            distances = np.linalg.norm(self.positions - center, axis=1)
            capsule_atoms = np.where(distances < capsule_radius)[0]
            
            if len(capsule_atoms) < 10:
                continue  # Skip if too few atoms
                
            # Shell atoms (Type 3)
            shell_atoms = capsule_atoms[:len(capsule_atoms)//2]
            self.atom_types[shell_atoms] = 3
            self.masses[shell_atoms] = 1.2  # Heavier shell
            
            # Healing agent inside capsule (Type 4)
            core_atoms = capsule_atoms[len(capsule_atoms)//2:]
            self.atom_types[core_atoms] = 4
            self.masses[core_atoms] = 0.8  # Lighter healing agent
    
    def _initialize_hydrogel(self) -> None:
        """Initialize a hydrogel material with water and polymer network"""
        # Water molecules (Type 0)
        water_fraction = 0.7
        n_water = int(self.config.n_atoms * water_fraction)
        self.atom_types[:n_water] = 0
        self.masses[:n_water] = 0.6  # Lighter water molecules
        
        # Polymer network (Type 1 and 2)
        polymer_atoms = np.arange(n_water, self.config.n_atoms)
        self.atom_types[polymer_atoms] = 1
        self.masses[polymer_atoms] = 1.0
        
        # Add crosslinking points (Type 2)
        n_crosslinks = int(len(polymer_atoms) * 0.2)
        crosslink_indices = np.random.choice(polymer_atoms, size=n_crosslinks, replace=False)
        self.atom_types[crosslink_indices] = 2
        
        # Add some stimuli-responsive elements (Type 5)
        n_responsive = int(len(polymer_atoms) * 0.1)
        responsive_indices = np.random.choice(
            polymer_atoms, 
            size=n_responsive, 
            replace=False
        )
        self.atom_types[responsive_indices] = 5
        self.masses[responsive_indices] = 1.2
    
    def _initialize_nanocomposite(self) -> None:
        """Initialize a nanocomposite material with nanoparticles in polymer matrix"""
        # Polymer matrix (Type 1)
        self.atom_types[:] = 1
        self.masses[:] = 1.0
        
        # Add nanoparticles (Type 6)
        n_particles = 5
        particle_radius = 0.8
        atoms_per_particle = self.config.n_atoms // (n_particles * 4)
        
        atoms_used = 0
        for i in range(n_particles):
            center = np.random.rand(3) * (np.array(self.config.box_size) - 2*particle_radius) + particle_radius
            
            distances = np.linalg.norm(self.positions - center, axis=1)
            sorted_indices = np.argsort(distances)
            
            particle_atoms = sorted_indices[:atoms_per_particle]
            self.atom_types[particle_atoms] = 6
            self.masses[particle_atoms] = 2.0  # Heavier nanoparticles
            
            # Interface atoms with functionalization (Type 7)
            interface_atoms = sorted_indices[atoms_per_particle:atoms_per_particle*2]
            self.atom_types[interface_atoms] = 7
            self.masses[interface_atoms] = 1.5
            
            atoms_used += atoms_per_particle * 2
    
    def _setup_molecular_topology(self) -> None:
        """Set up bonds and angles based on molecular structure"""
        self.bonds = []
        self.angles = []
        
        # Create bonds based on distance criteria
        bond_cutoff = 0.2  # nm
        for i in range(self.config.n_atoms):
            for j in range(i+1, self.config.n_atoms):
                r_ij = np.linalg.norm(self.positions[j] - self.positions[i])
                
                # Connect atoms that are close enough
                if r_ij < bond_cutoff:
                    self.bonds.append((i, j))
        
        # Create angles from bonds
        bond_map = {}
        for a, b in self.bonds:
            if a not in bond_map:
                bond_map[a] = []
            if b not in bond_map:
                bond_map[b] = []
            
            bond_map[a].append(b)
            bond_map[b].append(a)
        
        # Find all angles (i-j-k where i is bonded to j and j is bonded to k)
        for j in bond_map:
            neighbors = bond_map[j]
            for i_idx, i in enumerate(neighbors):
                for k in neighbors[i_idx+1:]:
                    # Avoid linear angles
                    if i != k:
                        self.angles.append((i, j, k))
    
    def calculate_forces(self) -> None:
        """Calculate forces between atoms with different potentials"""
        # Reset forces and energy
        self.forces.fill(0.0)
        self.energies = {
            'kinetic': 0.0,
            'potential': 0.0,
            'bond': 0.0,
            'angle': 0.0,
            'nonbonded': 0.0,
            'total': 0.0
        }
        
        # Non-bonded interactions (Lennard-Jones)
        for i in range(self.config.n_atoms):
            for j in range(i+1, self.config.n_atoms):
                # Skip if atoms are bonded
                if any((i, j) in self.bonds or (j, i) in self.bonds for bond in self.bonds):
                    continue
                
                r_ij = self.positions[j] - self.positions[i]
                r = np.linalg.norm(r_ij)
                
                # Apply cutoff
                if r < 2.5:
                    # Get parameters based on atom types
                    sigma, epsilon = self.force_field.get_pair_params(
                        self.atom_types[i], self.atom_types[j]
                    )
                    
                    # Calculate force and energy
                    force_magnitude, energy = self.force_field.lennard_jones(r, sigma, epsilon)
                    f_ij = force_magnitude * r_ij / r
                    
                    self.forces[i] += f_ij
                    self.forces[j] -= f_ij
                    self.energies['nonbonded'] += energy
        
        # Bond forces (Harmonic or Morse potential)
        for a, b in self.bonds:
            r_ij = self.positions[b] - self.positions[a]
            r = np.linalg.norm(r_ij)
            
            # Get bond parameters
            r0, k = self.force_field.get_bond_params(
                self.atom_types[a], self.atom_types[b]
            )
            
            # Use Morse potential for weaker bonds (e.g., crosslinks, type 2)
            if self.atom_types[a] == 2 or self.atom_types[b] == 2:
                D = 30.0  # Bond dissociation energy
                alpha = 4.0  # Controls width of potential well
                force_magnitude, energy = self.force_field.morse_potential(r, r0, D, alpha)
            else:
                force_magnitude, energy = self.force_field.harmonic_bond(r, r0, k)
            
            f_ij = force_magnitude * r_ij / r
            
            self.forces[a] += f_ij
            self.forces[b] -= f_ij
            self.energies['bond'] += energy
        
        # Angle forces
        for i, j, k in self.angles:
            # Get vectors for the angle
            r_ij = self.positions[i] - self.positions[j]
            r_kj = self.positions[k] - self.positions[j]
            
            # Normalize vectors
            r_ij_norm = r_ij / np.linalg.norm(r_ij)
            r_kj_norm = r_kj / np.linalg.norm(r_kj)
            
            # Calculate angle
            cos_angle = np.dot(r_ij_norm, r_kj_norm)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Avoid numerical errors
            angle = np.arccos(cos_angle)
            
            # Get angle parameters
            theta0, k = self.force_field.get_angle_params(
                self.atom_types[i], self.atom_types[j], self.atom_types[k]
            )
            
            # Calculate force and energy
            force_magnitude, energy = self.force_field.harmonic_angle(angle, theta0, k)
            
            # Forces on each atom (simplified)
            f_i = force_magnitude * np.cross(np.cross(r_ij, r_kj), r_ij) / np.linalg.norm(r_ij)
            f_k = force_magnitude * np.cross(np.cross(r_kj, r_ij), r_kj) / np.linalg.norm(r_kj)
            f_j = -(f_i + f_k)
            
            self.forces[i] += f_i
            self.forces[j] += f_j
            self.forces[k] += f_k
            self.energies['angle'] += energy
        
        # Special forces for self-healing mechanisms
        self._calculate_healing_forces()
        
        # Calculate kinetic energy
        self.energies['kinetic'] = 0.5 * np.sum(self.masses[:, np.newaxis] * self.velocities**2)
        
        # Total potential energy
        self.energies['potential'] = (
            self.energies['bond'] + 
            self.energies['angle'] + 
            self.energies['nonbonded']
        )
        
        # Total energy
        self.energies['total'] = self.energies['kinetic'] + self.energies['potential']
    
    def _calculate_healing_forces(self) -> None:
        """Calculate special forces for self-healing mechanisms"""
        # Healing agent attraction to damage sites
        healing_agent_indices = np.where(
            (self.atom_types == 4) | (self.atom_types == 5)
        )[0]
        
        for damage_site in self.damage_sites:
            a, b = damage_site
            if b == -1:  # Single atom damage
                damage_pos = self.positions[a]
            else:  # Bond damage
                damage_pos = (self.positions[a] + self.positions[b]) / 2
                
            for i in healing_agent_indices:
                r_ij = damage_pos - self.positions[i]
                r = np.linalg.norm(r_ij)
                
                # Attraction within a certain range
                if 0.5 < r < 2.0:
                    # Gaussian-like attraction
                    f_magnitude = 2.0 * np.exp(-(r-1.0)**2/0.5)
                    f_ij = f_magnitude * r_ij / r
                    
                    self.forces[i] += f_ij
        
        # Special forces for crosslinking points
        crosslink_indices = np.where(self.atom_types == 2)[0]
        for i in crosslink_indices:
            for j in crosslink_indices:
                if i != j:
                    r_ij = self.positions[j] - self.positions[i]
                    r = np.linalg.norm(r_ij)
                    
                    # Attraction in a specific range
                    if 0.3 < r < 0.8:
                        f_magnitude = 1.5 * np.exp(-(r-0.5)**2/0.1)
                        f_ij = f_magnitude * r_ij / r
                        
                        self.forces[i] += f_ij
                        self.forces[j] -= f_ij
    
    def apply_damage(self, damage_type: str = "crack", intensity: float = 0.5, location: Optional[np.ndarray] = None) -> None:
        """Apply different types of damage to the material"""
        logger.info(f"Applying {damage_type} damage with intensity {intensity}")
        
        if damage_type == "crack":
            # Simulate a crack by breaking bonds
            n_bonds_to_break = int(len(self.bonds) * intensity * 0.2)
            
            if location is None:
                # Random crack
                bonds_to_break = np.random.choice(len(self.bonds), n_bonds_to_break, replace=False)
            else:
                # Crack at specific location
                bond_distances = []
                for idx, (a, b) in enumerate(self.bonds):
                    bond_center = (self.positions[a] + self.positions[b]) / 2
                    dist = np.linalg.norm(bond_center - location)
                    bond_distances.append((idx, dist))
                
                # Sort by distance to location
                bond_distances.sort(key=lambda x: x[1])
                bonds_to_break = [idx for idx, _ in bond_distances[:n_bonds_to_break]]
            
            # Record damage sites and remove broken bonds
            for idx in bonds_to_break:
                if idx < len(self.bonds):
                    a, b = self.bonds[idx]
                    self.damage_sites.append((a, b))
            
            # Remove broken bonds
            self.bonds = [bond for i, bond in enumerate(self.bonds) if i not in bonds_to_break]
            
            # Update angles after bond breaking
            self._update_angles_after_damage()
            
        elif damage_type == "impact":
            # Simulate an impact at a specific point
            if location is None:
                impact_center = np.random.rand(3) * self.config.box_size
            else:
                impact_center = location
                
            impact_radius = intensity * 2.0
            
            # Find atoms affected by the impact
            distances = np.linalg.norm(self.positions - impact_center, axis=1)
            affected_atoms = np.where(distances < impact_radius)[0]
            
            # Displace affected atoms
            displacement_magnitude = intensity * 0.5
            for atom in affected_atoms:
                # Direction away from impact
                direction = self.positions[atom] - impact_center
                if np.linalg.norm(direction) > 0:
                    direction = direction / np.linalg.norm(direction)
                else:
                    direction = np.random.randn(3)
                    direction = direction / np.linalg.norm(direction)
                
                # Add displacement
                self.positions[atom] += direction * displacement_magnitude
                
                # Record damage
                self.damage_sites.append((atom, -1))
                
            # Break bonds near impact center
            bonds_to_break = []
            for idx, (a, b) in enumerate(self.bonds):
                bond_center = (self.positions[a] + self.positions[b]) / 2
                dist = np.linalg.norm(bond_center - impact_center)
                if dist < impact_radius * 0.7:
                    bonds_to_break.append(idx)
                    self.damage_sites.append((a, b))
            
            # Remove broken bonds
            self.bonds = [bond for i, bond in enumerate(self.bonds) if i not in bonds_to_break]
            
            # Update angles after impact
            self._update_angles_after_damage()
            
        elif damage_type == "thermal":
            # Thermal damage - weaken bonds and increase local temperature
            if location is None:
                heat_center = np.random.rand(3) * self.config.box_size
            else:
                heat_center = location
                
            heat_radius = intensity * 3.0
            
            # Find atoms in heated region
            distances = np.linalg.norm(self.positions - heat_center, axis=1)
            heated_atoms = np.where(distances < heat_radius)[0]
            
            # Increase velocity (temperature) of affected atoms
            temp_increase_factor = 1.0 + intensity * 2.0
            for atom in heated_atoms:
                self.velocities[atom] *= temp_increase_factor
                self.damage_sites.append((atom, -1))
            
            # Weaken some bonds in heated region
            bonds_to_weaken = []
            for idx, (a, b) in enumerate(self.bonds):
                bond_center = (self.positions[a] + self.positions[b]) / 2
                dist = np.linalg.norm(bond_center - heat_center)
                if dist < heat_radius and np.random.random() < intensity * 0.5:
                    bonds_to_weaken.append(idx)
                    self.damage_sites.append((a, b))
            
            # For simplicity, just break the weakened bonds
            self.bonds = [bond for i, bond in enumerate(self.bonds) if i not in bonds_to_weaken]
            
        elif damage_type == "chemical":
            # Chemical damage - change atom types and break specific bonds
            if location is None:
                chem_center = np.random.rand(3) * self.config.box_size
            else:
                chem_center = location
                
            chem_radius = intensity * 2.5
            
            # Find atoms in affected region
            distances = np.linalg.norm(self.positions - chem_center, axis=1)
            affected_atoms = np.where(distances < chem_radius)[0]
            
            # Change atom types to simulate chemical modification
            for atom in affected_atoms:
                if self.atom_types[atom] in [1, 2]:  # Only affect polymer atoms
                    # Change to a "degraded" type (type 8)
                    self.atom_types[atom] = 8
                    self.damage_sites.append((atom, -1))
            
            # Break bonds between modified atoms
            bonds_to_break = []
            for idx, (a, b) in enumerate(self.bonds):
                if (self.atom_types[a] == 8 or self.atom_types[b] == 8) and np.random.random() < 0.7:
                    bonds_to_break.append(idx)
                    self.damage_sites.append((a, b))
            
            # Remove broken bonds
            self.bonds = [bond for i, bond in enumerate(self.bonds) if i not in bonds_to_break]
        
        else:
            raise ValueError(f"Unknown damage type: {damage_type}")
        
        logger.info(f"Damage applied: {len(self.damage_sites)} damage sites created")
    
    def _update_angles_after_damage(self) -> None:
        """Update angle list after bonds have been broken"""
        # Rebuild bond map
        bond_map = {}
        for a, b in self.bonds:
            if a not in bond_map:
                bond_map[a] = []
            if b not in bond_map:
                bond_map[b] = []
            
            bond_map[a].append(b)
            bond_map[b].append(a)
        
        # Rebuild angles
        self.angles = []
        for j in bond_map:
            neighbors = bond_map[j]
            for i_idx, i in enumerate(neighbors):
                for k in neighbors[i_idx+1:]:
                    if i != k:
                        self.angles.append((i, j, k))
    
    def step(self) -> None:
        """Perform one time step of molecular dynamics simulation"""
        dt = self.config.time_step
        
        # Calculate initial forces
        self.calculate_forces()
        
        # Velocity Verlet integration (first half)
        self.positions += self.velocities * dt + 0.5 * self.forces * dt**2
        
        old_velocities = self.velocities.copy()
        self.velocities += 0.5 * self.forces * dt
        
        # Apply periodic boundary conditions
        self.positions = np.mod(self.positions, self.config.box_size)
        
        # Calculate forces with new positions
        old_forces = self.forces.copy()
        self.calculate_forces()
        
        # Velocity Verlet integration (second half)
        self.velocities += 0.5 * self.forces * dt
        
        # Temperature control using Berendsen thermostat
        if self.config.temperature > 0:
            # Calculate current temperature
            kinetic_energy = 0.5 * np.sum(self.masses[:, np.newaxis] * self.velocities**2)
            degrees_of_freedom = 3 * self.config.n_atoms - 3  # Subtract center of mass motion
            current_temp = 2 * kinetic_energy / (degrees_of_freedom * 0.00831)
            
            # Scale velocities to maintain target temperature
            tau = 0.1  # coupling constant (ps)
            lambda_factor = np.sqrt(1 + (dt/tau) * (self.config.temperature/current_temp - 1))
            self.velocities *= lambda_factor
        
        # Pressure control (basic implementation of Berendsen barostat)
        if hasattr(self.config, 'pressure') and self.config.pressure > 0:
            # Calculate stress tensor
            self._calculate_stress_tensor()
            
            # Current pressure (trace of stress tensor divided by 3)
            current_pressure = np.trace(self.stress_tensor) / 3.0
            
            # Scale coordinates and box size
            tau_p = 1.0  # pressure coupling constant
            compressibility = 4.5e-5  # isothermal compressibility of water
            beta = 1.0 / (self.config.temperature * 0.00831)
            
            # Scaling factor
            mu = 1 - (dt/tau_p) * compressibility * (self.config.pressure - current_pressure)
            
            # Apply scaling to positions and box size
            if 0.95 < mu < 1.05:  # Limit scaling to prevent instability
                self.positions *= mu**(1/3)
                self.config.box_size = [bs * mu**(1/3) for bs in self.config.box_size]
    
    def _calculate_stress_tensor(self) -> None:
        """Calculate the stress tensor of the system"""
        volume = np.prod(self.config.box_size)
        self.stress_tensor = np.zeros((3, 3))
        
        # Kinetic contribution
        for i in range(self.config.n_atoms):
            v = self.velocities[i]
            m = self.masses[i]
            self.stress_tensor += m * np.outer(v, v)
        
        # Virial contribution from non-bonded forces (simplified)
        for i in range(self.config.n_atoms):
            for j in range(i+1, self.config.n_atoms):
                r_ij = self.positions[j] - self.positions[i]
                r = np.linalg.norm(r_ij)
                
                # Skip if atoms are too far apart
                if r >= 2.5:
                    continue
                
                # Get parameters and calculate force
                sigma, epsilon = self.force_field.get_pair_params(
                    self.atom_types[i], self.atom_types[j]
                )
                force_magnitude, _ = self.force_field.lennard_jones(r, sigma, epsilon)
                
                # Contribution to stress tensor
                self.stress_tensor += force_magnitude * np.outer(r_ij, r_ij) / r
        
        # Convert to pressure units
        self.stress_tensor /= volume
    
    def run_simulation(self, n_steps: Optional[int] = None, healing_agent: Optional[Any] = None) -> Dict:
        """Run the molecular dynamics simulation for n_steps"""
        if n_steps is None:
            n_steps = self.config.n_steps
            
        logger.info(f"Starting MD simulation for {n_steps} steps")
        
        # Data collections
        trajectories = []
        velocities = []
        bond_data = []
        energy_data = []
        atom_type_changes = []
        
        # Run simulation
        for step in tqdm(range(n_steps), desc="MD Simulation"):
            # Store current state
            if step % self.config.save_interval == 0:
                trajectories.append(self.positions.copy())
                velocities.append(self.velocities.copy())
                bond_data.append(self.bonds.copy())
                energy_data.append(self.energies.copy())
                atom_type_changes.append(self.atom_types.copy())
            
            # Perform time step
            self.step()
            
            # Apply healing agent if provided
            if healing_agent is not None and step > n_steps // 4:
                # Don't apply healing every step to save computation
                if step % 50 == 0:
                    self.apply_healing_agent(healing_agent)
        
        logger.info(f"Simulation completed with {len(self.bonds)} final bonds")
        
        return {
            "trajectories": np.array(trajectories),
            "velocities": np.array(velocities),
            "bond_data": bond_data,
            "energy_data": energy_data,
            "atom_type_changes": atom_type_changes,
            "final_state": {
                "positions": self.positions.copy(),
                "velocities": self.velocities.copy(),
                "atom_types": self.atom_types.copy(),
                "bonds": self.bonds.copy(),
                "angles": self.angles.copy(),
                "energies": self.energies.copy()
            }
        }
    
    def apply_healing_agent(self, healing_agent) -> None:
        """Apply the ML-based healing agent to repair damage"""
        # Convert current state to tensor for the ML model
        state_tensor = self.get_state_tensor()
        
        # Get healing predictions from the model
        healing_actions = healing_agent.predict_healing_actions(state_tensor)
        
        # Apply the predicted healing actions
        for action in healing_actions:
            action_type, params = action
            
            if action_type == "form_bond":
                a, b = params
                if 0 <= a < self.config.n_atoms and 0 <= b < self.config.n_atoms:
                    # Check if atoms are close enough
                    r_ab = np.linalg.norm(self.positions[a] - self.positions[b])
                    r0, _ = self.force_field.get_bond_params(
                        self.atom_types[a], self.atom_types[b]
                    )
                    
                    if r_ab < r0 * 1.5:  # Allow slightly longer bonds for healing
                        # Check if bond already exists
                        if (a, b) not in self.bonds and (b, a) not in self.bonds:
                            self.bonds.append((a, b))
                            
                            # Update angles that might be created by this new bond
                            self._update_angles_after_damage()
                            
                            logger.debug(f"Bond formed between atoms {a} and {b}")
            
            elif action_type == "activate_healing_agent":
                center, radius = params
                # Find healing agent atoms (type 4) near the damage
                distances = np.linalg.norm(self.positions - center, axis=1)
                healing_atoms = np.where(
                    (distances < radius) & (self.atom_types == 4)
                )[0]
                
                # Change their type to activated healing agent (type 5)
                self.atom_types[healing_atoms] = 5
                logger.debug(f"Activated {len(healing_atoms)} healing agent atoms")
            
            elif action_type == "change_atom_type":
                atom_idx, new_type = params
                if 0 <= atom_idx < self.config.n_atoms:
                    self.atom_types[atom_idx] = new_type
                    logger.debug(f"Changed atom {atom_idx} to type {new_type}")
    
    def get_state_tensor(self) -> torch.Tensor:
        """Convert current system state to tensor representation for ML models"""
        # Normalized positions
        pos_normalized = self.positions / np.array(self.config.box_size)
        
        # One-hot encoding for atom types (assuming max 10 types)
        max_type = 10
        type_onehot = np.zeros((self.config.n_atoms, max_type))
        for i in range(self.config.n_atoms):
            type_id = min(self.atom_types[i], max_type - 1)  # Clip to prevent index errors
            type_onehot[i, type_id] = 1
        
        # Local density feature
        density_feature = np.zeros((self.config.n_atoms, 1))
        cutoff = 1.0  # nm
        for i in range(self.config.n_atoms):
            neighbors = 0
            for j in range(self.config.n_atoms):
                if i != j:
                    r_ij = np.linalg.norm(self.positions[i] - self.positions[j])
                    if r_ij < cutoff:
                        neighbors += 1
            density_feature[i, 0] = neighbors / (4/3 * np.pi * cutoff**3)
        
        # Bond connectivity (simplified for transformer input)
        bond_feature = np.zeros((self.config.n_atoms, 1))
        for a, b in self.bonds:
            bond_feature[a, 0] += 1
            bond_feature[b, 0] += 1
        
        # Damage proximity feature
        damage_feature = np.zeros((self.config.n_atoms, 1))
        for a, b in self.damage_sites:
            if b == -1:  # Single atom damage
                damage_pos = self.positions[a]
            else:  # Bond damage
                damage_pos = (self.positions[a] + self.positions[b]) / 2
            
            for i in range(self.config.n_atoms):
                r_i = np.linalg.norm(self.positions[i] - damage_pos)
                # Gaussian influence of damage site
                damage_feature[i, 0] += np.exp(-(r_i/0.5)**2)
        
        # Combine features
        features = np.concatenate([
            pos_normalized,          # 3 features
            type_onehot,             # max_type features
            density_feature,         # 1 feature
            bond_feature,            # 1 feature
            damage_feature,          # 1 feature
            self.velocities / 5.0,   # 3 features (normalized)
        ], axis=1)
        
        return torch.FloatTensor(features)


class MDTrajectoryDataset(Dataset):
    """Dataset for training ML models on MD trajectory data"""
    
    def __init__(self, trajectories, velocities, atom_types, bonds, damage_sites=None):
        self.trajectories = trajectories
        self.velocities = velocities
        self.atom_types = atom_types
        self.bonds = bonds
        self.damage_sites = damage_sites
        
        # Generate graph adjacency data for GNN
        self.adjacency_data = self._generate_adjacency_data()
    
    def __len__(self):
        return len(self.trajectories) - 1
    
    def __getitem__(self, idx):
        # Current and next state data
        current_state = self.trajectories[idx]
        next_state = self.trajectories[idx + 1]
        
        current_vel = self.velocities[idx]
        next_vel = self.velocities[idx + 1]
        
        # Convert to tensors
        current_pos = torch.FloatTensor(current_state)
        target_pos = torch.FloatTensor(next_state)
        
        current_velocity = torch.FloatTensor(current_vel)
        target_velocity = torch.FloatTensor(next_vel)
        
        # Get adjacency matrix for this frame
        adj_matrix = self.adjacency_data[idx]
        
        return {
            "current_pos": current_pos,
            "target_pos": target_pos,
            "current_vel": current_velocity,
            "target_vel": target_velocity,
            "atom_types": torch.LongTensor(self.atom_types),
            "bonds": self.bonds[idx],
            "adj_matrix": adj_matrix
        }
    
    def _generate_adjacency_data(self):
        """Generate adjacency matrices for graph networks"""
        adjacency_data = []
        
        for frame_idx, bonds in enumerate(self.bonds):
            n_atoms = self.trajectories.shape[1]
            adj_matrix = torch.zeros(n_atoms, n_atoms)
            
            # Fill adjacency matrix based on bonds
            for a, b in bonds:
                adj_matrix[a, b] = 1
                adj_matrix[b, a] = 1
            
            adjacency_data.append(adj_matrix)
        
        return adjacency_data
    
    def get_edge_index(self, idx):
        """Get edge index representation for PyTorch Geometric"""
        bonds = self.bonds[idx]
        edge_index = torch.LongTensor([[a, b] for a, b in bonds] + [[b, a] for a, b in bonds]).t()
        return edge_index


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models"""
    
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """Add positional encoding to input tensor"""
        return x + self.pe[:, :x.size(1)]


class TransformerHealingAgent(nn.Module):
    """Transformer-based model for predicting healing actions"""
    
    def __init__(self, config):
        super(TransformerHealingAgent, self).__init__()
        self.config = config
        
        # Input dimensions
        d_model = config.model_dim
        input_dim = 3 + 10 + 1 + 1 + 1 + 3  # pos + type + density + bond + damage + vel
        
        # Input projections
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=config.n_heads,
            dim_feedforward=d_model * 4,
            dropout=config.dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, 
            num_layers=config.n_layers
        )
        
        # Output heads
        self.bond_predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid()
        )
        
        self.position_predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 3),
            nn.Tanh()  # Limit position adjustments
        )
        
        self.type_predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 10),  # Predict probability of each atom type
            nn.Softmax(dim=-1)
        )
        
        # Attention for damage detection
        self.damage_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
        
        self.damage_classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x, mask=None):
        """Forward pass through the model"""
        # x shape: [batch_size, n_atoms, features]
        batch_size, n_atoms, _ = x.shape
        
        # Project input to model dimension
        x = self.input_projection(x)
        
        # Apply positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoder
        if mask is None:
            encoder_output = self.transformer_encoder(x)
        else:
            encoder_output = self.transformer_encoder(x, src_key_padding_mask=mask)
        
        # Predict bond formation probabilities (n_atoms x n_atoms matrix)
        # For each atom pair, predict if they should form a bond
        bond_logits = torch.zeros(batch_size, n_atoms, n_atoms, device=x.device)
        
        for i in range(n_atoms):
            # Get embedding for atom i
            atom_i_emb = encoder_output[:, i, :].unsqueeze(1).repeat(1, n_atoms, 1)
            
            # Combine with all other atoms
            atom_pairs = torch.cat([atom_i_emb, encoder_output], dim=2)
            
            # Predict bond formation
            bond_probs_i = self.bond_predictor(atom_pairs).squeeze(-1)
            bond_logits[:, i, :] = bond_probs_i
        
        # Position adjustments
        pos_delta = self.position_predictor(encoder_output) * 0.1  # Scale adjustments
        
        # Atom type predictions
        type_probs = self.type_predictor(encoder_output)
        
        # Damage detection using attention
        damage_scores, _ = self.damage_attention(
            encoder_output, encoder_output, encoder_output
        )
        damage_probs = self.damage_classifier(damage_scores).squeeze(-1)
        
        return bond_logits, pos_delta, type_probs, damage_probs
    
    def predict_healing_actions(self, state_tensor):
        """Predict healing actions based on the current state"""
        self.eval()
        with torch.no_grad():
            # Make sure input has batch dimension
            if state_tensor.dim() == 2:
                state_tensor = state_tensor.unsqueeze(0)
                
            # Move to correct device
            state_tensor = state_tensor.to(self.config.device)
            
            # Forward pass
            bond_probs, pos_delta, type_probs, damage_probs = self.forward(state_tensor)
            
            # Process predictions
            bond_probs = bond_probs.squeeze(0).cpu().numpy()
            pos_delta = pos_delta.squeeze(0).cpu().numpy()
            type_probs = type_probs.squeeze(0).cpu().numpy()
            damage_probs = damage_probs.squeeze(0).cpu().numpy()
            
            # Find potential healing actions
            actions = []
            
            # 1. Bond formation actions
            for i in range(bond_probs.shape[0]):
                for j in range(i+1, bond_probs.shape[1]):
                    # High probability of bond formation
                    if bond_probs[i, j] > 0.8:
                        actions.append(("form_bond", (i, j)))
            
            # 2. Healing agent activation near damage
            # Find damage sites
            damage_sites = np.where(damage_probs > 0.7)[0]
            
            for damage_idx in damage_sites:
                center = state_tensor[0, damage_idx, :3].cpu().numpy()
                actions.append(("activate_healing_agent", (center, 1.0)))
            
            # 3. Atom type changes (e.g., activating healing agents)
            for i in range(type_probs.shape[0]):
                current_type = np.argmax(state_tensor[0, i, 3:13].cpu().numpy())
                predicted_type = np.argmax(type_probs[i])
                
                # If model predicts a type change
                if current_type != predicted_type and type_probs[i, predicted_type] > 0.8:
                    actions.append(("change_atom_type", (i, predicted_type)))
            
            # Limit number of actions per step
            if len(actions) > 10:
                actions = actions[:10]
                
            return actions


class GCNHealingAgent(nn.Module):
    """Graph Convolutional Network for molecular healing prediction"""
    
    def __init__(self, config, feature_dim=16):
        super(GCNHealingAgent, self).__init__()
        self.config = config
        
        # Node feature embedding
        self.node_embedding = nn.Linear(feature_dim, config.model_dim)
        
        # Graph convolutional layers
        self.conv1 = GCNConv(config.model_dim, config.model_dim)
        self.conv2 = GCNConv(config.model_dim, config.model_dim)
        self.conv3 = GCNConv(config.model_dim, config.model_dim)
        
        # Edge prediction layers
        self.edge_predictor = nn.Sequential(
            nn.Linear(config.model_dim * 2, config.model_dim),
            nn.ReLU(),
            nn.Linear(config.model_dim, 1),
            nn.Sigmoid()
        )
        
        # Node property prediction
        self.node_predictor = nn.Sequential(
            nn.Linear(config.model_dim, config.model_dim),
            nn.ReLU(),
            nn.Linear(config.model_dim, 10 + 3)  # Type (10) + position delta (3)
        )
        
        # Global readout
        self.global_predictor = nn.Sequential(
            nn.Linear(config.model_dim, config.model_dim // 2),
            nn.ReLU(),
            nn.Linear(config.model_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x, edge_index):
        """Forward pass through the GCN"""
        # Embed node features
        x = self.node_embedding(x)
        
        # Graph convolutions with residual connections
        x1 = F.relu(self.conv1(x, edge_index))
        x = x + self.dropout(x1)
        
        x2 = F.relu(self.conv2(x, edge_index))
        x = x + self.dropout(x2)
        
        x3 = F.relu(self.conv3(x, edge_index))
        x = x + self.dropout(x3)
        
        # Node predictions (atom type and position)
        node_preds = self.node_predictor(x)
        type_logits = node_preds[:, :10]
        pos_delta = node_preds[:, 10:]
        
        # Global readout for overall status
        global_feature = global_mean_pool(x, torch.zeros(x.size(0), dtype=torch.long, device=x.device))
        global_pred = self.global_predictor(global_feature)
        
        return x, type_logits, pos_delta, global_pred
    
    def predict_edge(self, node_features, i, j):
        """Predict if there should be an edge between nodes i and j"""
        # Concatenate node features
        edge_features = torch.cat([node_features[i], node_features[j]], dim=-1)
        return self.edge_predictor(edge_features)


class HybridHealingAgent(nn.Module):
    """Hybrid model combining transformer and GCN for self-healing prediction"""
    
    def __init__(self, config):
        super(HybridHealingAgent, self).__init__()
        self.config = config
        
        # Transformer component for global context
        self.transformer = TransformerHealingAgent(config)
        
        # GCN component for local structure
        self.gcn = GCNHealingAgent(config)
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(config.model_dim * 2, config.model_dim),
            nn.ReLU(),
            nn.Linear(config.model_dim, config.model_dim)
        )
        
        # Final prediction layers
        self.final_bond_predictor = nn.Sequential(
            nn.Linear(config.model_dim, 1),
            nn.Sigmoid()
        )
        
        self.final_type_predictor = nn.Sequential(
            nn.Linear(config.model_dim, 10),
            nn.Softmax(dim=-1)
        )
        
        self.final_pos_predictor = nn.Sequential(
            nn.Linear(config.model_dim, 3),
            nn.Tanh()
        )
    
    def forward(self, x, edge_index=None, adj_matrix=None):
        """Forward pass through the hybrid model"""
        # Get transformer features
        bond_logits_t, pos_delta_t, type_probs_t, damage_probs_t = self.transformer(x)
        
        # For GCN, we need to create edge_index if it's not provided
        if edge_index is None and adj_matrix is not None:
            edge_index = self._adj_to_edge_index(adj_matrix)
        
        # Get GCN features
        gcn_features, type_logits_g, pos_delta_g, global_pred_g = self.gcn(x.reshape(-1, x.size(-1)), edge_index)
        
        # Reshape GCN features to match transformer
        gcn_features = gcn_features.reshape(x.size(0), x.size(1), -1)
        
        # Fuse features
        fused_features = self.fusion(
            torch.cat([
                self.transformer.transformer_encoder(self.transformer.pos_encoder(self.transformer.input_projection(x))),
                gcn_features
            ], dim=-1)
        )
        
        # Final predictions
        final_bond_logits = torch.zeros_like(bond_logits_t)
        final_type_probs = self.final_type_predictor(fused_features)
        final_pos_delta = self.final_pos_predictor(fused_features)
        
        # Calculate pairwise bond probabilities
        n_atoms = x.size(1)
        for i in range(n_atoms):
            for j in range(i+1, n_atoms):
                # Get embeddings for the pair
                pair_emb = torch.cat([fused_features[:, i], fused_features[:, j]], dim=-1)
                bond_prob = self.final_bond_predictor(pair_emb)
                
                # Update bond logits
                final_bond_logits[:, i, j] = bond_prob
                final_bond_logits[:, j, i] = bond_prob
        
        return final_bond_logits, final_pos_delta, final_type_probs, damage_probs_t
    
    def _adj_to_edge_index(self, adj_matrix):
        """Convert adjacency matrix to edge index format"""
        edges = torch.nonzero(adj_matrix, as_tuple=True)
        edge_index = torch.stack(edges, dim=0)
        return edge_index
    
    def predict_healing_actions(self, state_tensor):
        """Predict healing actions using the hybrid model"""
        # This can leverage both transformer and GCN predictions
        # For simplicity, we'll use the transformer's prediction method
        return self.transformer.predict_healing_actions(state_tensor)


def train_healing_agent(model, train_loader, val_loader, config):
    """Train the healing agent model"""
    logger.info(f"Training {config.model_type} model for {config.epochs} epochs")
    
    # Setup training
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Loss functions
    bond_criterion = nn.BCELoss()
    type_criterion = nn.CrossEntropyLoss()
    pos_criterion = nn.MSELoss()
    damage_criterion = nn.BCELoss()
    
    # Training metrics
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    # Create model save directory
    os.makedirs(config.model_save_path, exist_ok=True)
    
    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}"):
            optimizer.zero_grad()
            
            # Move data to device
            current_pos = batch["current_pos"].to(config.device)
            target_pos = batch["target_pos"].to(config.device)
            current_vel = batch["current_vel"].to(config.device)
            target_vel = batch["target_vel"].to(config.device)
            atom_types = batch["atom_types"].to(config.device)
            
            # Create input tensor from position, velocity, and atom type
            input_tensor = torch.cat([
                current_pos,
                current_vel,
                F.one_hot(atom_types, num_classes=10).float()
            ], dim=-1)
            
            # Forward pass - different for model types
            if config.model_type == "transformer":
                bond_logits, pos_delta, type_logits, damage_logits = model(input_tensor)
                
                # Create bond target tensor
                bond_targets = torch.zeros_like(bond_logits)
                for i, bonds in enumerate(batch["bonds"]):
                    for a, b in bonds:
                        if a < bond_targets.size(1) and b < bond_targets.size(1):
                            bond_targets[i, a, b] = 1.0
                            bond_targets[i, b, a] = 1.0
                
                # Position target
                pos_targets = target_pos - current_pos
                
                # Calculate losses
                bond_loss = bond_criterion(bond_logits, bond_targets)
                pos_loss = pos_criterion(pos_delta, pos_targets)
                type_loss = type_criterion(
                    type_logits.reshape(-1, 10),
                    atom_types.reshape(-1)
                )
                
                # Simple damage target (assuming no damage for training)
                damage_targets = torch.zeros_like(damage_logits)
                damage_loss = damage_criterion(damage_logits, damage_targets)
                
                # Total loss
                loss = bond_loss + pos_loss + type_loss + damage_loss
                
            else:
                # For GCN and hybrid models
                # Create edge index from bonds
                edge_indices = []
                for bonds in batch["bonds"]:
                    edges = []
                    for a, b in bonds:
                        edges.append((a, b))
                        edges.append((b, a))  # Add reverse edge
                    
                    if not edges:
                        # If no bonds, add self-loops
                        edges = [(i, i) for i in range(current_pos.size(1))]
                    
                    edge_idx = torch.tensor(edges, dtype=torch.long).t()
                    edge_indices.append(edge_idx)
                
                # Placeholder for batched graph data
                # In practice, you'd use PyTorch Geometric's collate
                loss = 0.0
                for i in range(len(batch["current_pos"])):
                    sample_input = input_tensor[i].unsqueeze(0)
                    sample_edge_idx = edge_indices[i].to(config.device)
                    
                    # Forward pass
                    if config.model_type == "gcn":
                        node_features, type_preds, pos_preds, global_pred = model(
                            sample_input.reshape(-1, sample_input.size(-1)),
                            sample_edge_idx
                        )
                        
                        # Losses
                        type_loss = type_criterion(
                            type_preds,
                            atom_types[i]
                        )
                        
                        pos_loss = pos_criterion(
                            pos_preds,
                            (target_pos[i] - current_pos[i])
                        )
                        
                        sample_loss = type_loss + pos_loss
                        
                    elif config.model_type == "hybrid":
                        bond_logits, pos_delta, type_logits, damage_logits = model(
                            sample_input,
                            sample_edge_idx
                        )
                        
                        # Create targets
                        bond_targets = torch.zeros_like(bond_logits)
                        for a, b in batch["bonds"][i]:
                            if a < bond_targets.size(1) and b < bond_targets.size(1):
                                bond_targets[0, a, b] = 1.0
                                bond_targets[0, b, a] = 1.0
                        
                        # Losses
                        bond_loss = bond_criterion(bond_logits, bond_targets)
                        pos_loss = pos_criterion(pos_delta, target_pos[i] - current_pos[i])
                        type_loss = type_criterion(
                            type_logits.reshape(-1, 10),
                            atom_types[i].reshape(-1)
                        )
                        
                        sample_loss = bond_loss + pos_loss + type_loss
                    
                    loss += sample_loss
                
                loss /= len(batch["current_pos"])
            
            # Backpropagation
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            epoch_loss += loss.item()
        
        # Calculate average loss
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        val_loss = validate_model(model, val_loader, config)
        val_losses.append(val_loss)
        
        # Learning rate scheduler
        scheduler.step(val_loss)
        
        # Print progress
        logger.info(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {val_loss:.4f}")
        
        # Save if best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                model.state_dict(),
                os.path.join(config.model_save_path, f"best_{config.model_type}_agent.pt")
            )
            logger.info(f"Saved new best model with validation loss {val_loss:.4f}")
        
        # Always save last model
        torch.save(
            model.state_dict(),
            os.path.join(config.model_save_path, f"last_{config.model_type}_agent.pt")
        )
    
    # Training complete
    logger.info(f"Training completed with best validation loss: {best_val_loss:.4f}")
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss
    }


def validate_model(model, val_loader, config):
    """Validate the model on the validation dataset"""
    model.eval()
    
    # Loss functions
    bond_criterion = nn.BCELoss()
    type_criterion = nn.CrossEntropyLoss()
    pos_criterion = nn.MSELoss()
    damage_criterion = nn.BCELoss()
    
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in val_loader:
            # Move data to device
            current_pos = batch["current_pos"].to(config.device)
            target_pos = batch["target_pos"].to(config.device)
            current_vel = batch["current_vel"].to(config.device)
            atom_types = batch["atom_types"].to(config.device)
            
            # Create input tensor
            input_tensor = torch.cat([
                current_pos,
                current_vel,
                F.one_hot(atom_types, num_classes=10).float()
            ], dim=-1)
            
            # Forward pass
            if config.model_type == "transformer":
                bond_logits, pos_delta, type_logits, damage_logits = model(input_tensor)
                
                # Create bond target tensor
                bond_targets = torch.zeros_like(bond_logits)
                for i, bonds in enumerate(batch["bonds"]):
                    for a, b in bonds:
                        if a < bond_targets.size(1) and b < bond_targets.size(1):
                            bond_targets[i, a, b] = 1.0
                            bond_targets[i, b, a] = 1.0
                
                # Position target
                pos_targets = target_pos - current_pos
                
                # Calculate losses
                bond_loss = bond_criterion(bond_logits, bond_targets)
                pos_loss = pos_criterion(pos_delta, pos_targets)
                type_loss = type_criterion(
                    type_logits.reshape(-1, 10),
                    atom_types.reshape(-1)
                )
                
                # Simple damage target (assuming no damage for validation)
                damage_targets = torch.zeros_like(damage_logits)
                damage_loss = damage_criterion(damage_logits, damage_targets)
                
                # Total loss
                loss = bond_loss + pos_loss + type_loss + damage_loss
                
            else:
                # GCN and hybrid validation
                # Simplified for demonstration
                loss = 0.0
                # Similar validation code as in training but with batched graph data
            
            total_loss += loss.item()
    
    return total_loss / len(val_loader)


class MaterialAnalyzer:
    """Class for analyzing material properties from simulation data"""
    
    def __init__(self, config):
        self.config = config
    
    def analyze_material_integrity(self, bonds, initial_bonds):
        """Calculate material integrity based on bonds"""
        initial_count = len(initial_bonds)
        current_count = len(bonds)
        return (current_count / initial_count) * 100 if initial_count > 0 else 0
    
    def calculate_bond_energy(self, positions, bonds, atom_types, force_field):
        """Calculate total bond energy"""
        energy = 0.0
        
        for a, b in bonds:
            if a < len(positions) and b < len(positions):
                r_ij = positions[b] - positions[a]
                r = np.linalg.norm(r_ij)
                
                # Get parameters for this bond
                r0, k = force_field.get_bond_params(atom_types[a], atom_types[b])
                
                # Calculate energy
                energy += 0.5 * k * (r - r0)**2
        
        return energy
    
    def analyze_mechanical_properties(self, positions, bonds, atom_types):
        """Estimate basic mechanical properties from the molecular structure"""
        # Count crosslinks and calculate crosslink density
        crosslink_atoms = np.where(atom_types == 2)[0]
        crosslink_density = len(crosslink_atoms) / len(atom_types)
        
        # Calculate average coordination number (bonds per atom)
        bond_count = {}
        for a, b in bonds:
            bond_count[a] = bond_count.get(a, 0) + 1
            bond_count[b] = bond_count.get(b, 0) + 1
        
        avg_coordination = np.mean(list(bond_count.values())) if bond_count else 0
        
        # Estimate network connectivity
        # Higher values suggest better mechanical properties
        network_connectivity = crosslink_density * avg_coordination
        
        # Simple estimate of Young's modulus based on connectivity
        # This is a very rough approximation
        estimated_youngs_modulus = 10.0 * network_connectivity
        
        return {
            'crosslink_density': crosslink_density,
            'avg_coordination': avg_coordination,
            'network_connectivity': network_connectivity,
            'estimated_youngs_modulus': estimated_youngs_modulus
        }
    
    def analyze_healing_efficiency(self, initial_bonds, damaged_bonds, healed_bonds):
        """Calculate healing efficiency"""
        initial_count = len(initial_bonds)
        damaged_count = len(damaged_bonds)
        healed_count = len(healed_bonds)
        
        damage_percentage = 100 * (1 - damaged_count / initial_count)
        healed_percentage = 100 * healed_count / initial_count
        
        # Healing efficiency (how much of the damage was recovered)
        if initial_count == damaged_count:  # No damage
            healing_efficiency = 100.0
        else:
            healing_efficiency = 100 * (healed_count - damaged_count) / (initial_count - damaged_count)
        
        return {
            'damage_percentage': damage_percentage,
            'healed_percentage': healed_percentage,
            'healing_efficiency': healing_efficiency
        }
    
    def analyze_structural_recovery(self, initial_positions, damaged_positions, healed_positions):
        """Analyze structural recovery using RMSD"""
        # Root mean square deviation between structures
        rmsd_damaged = np.sqrt(np.mean(np.sum((damaged_positions - initial_positions)**2, axis=1)))
        rmsd_healed = np.sqrt(np.mean(np.sum((healed_positions - initial_positions)**2, axis=1)))
        
        # Structural recovery percentage
        if rmsd_damaged == 0:  # No damage
            structural_recovery = 100.0
        else:
            structural_recovery = 100 * (1 - rmsd_healed / rmsd_damaged)
        
        return {
            'rmsd_damaged': rmsd_damaged,
            'rmsd_healed': rmsd_healed,
            'structural_recovery': structural_recovery
        }


class Visualizer:
    """Class for visualizing material structures and properties"""
    
    def __init__(self, config):
        self.config = config
        # Set a consistent style
        sns.set(style="whitegrid")
        
        # Atom type color mapping
        self.atom_colors = {
            0: 'blue',      # Default atoms
            1: 'green',     # Polymer chain
            2: 'red',       # Crosslinking points
            3: 'purple',    # Capsule shell
            4: 'orange',    # Healing agent
            5: 'cyan',      # Activated healing agent
            6: 'brown',     # Nanoparticles
            7: 'magenta',   # Interface atoms
            8: 'gray',      # Damaged atoms
            9: 'yellow'     # Other
        }
    
    def visualize_material(self, positions, atom_types, bonds=None, damage_sites=None, highlight=None, filename=None):
        """Visualize the material state in 3D"""
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot atoms
        for i, (pos, atom_type) in enumerate(zip(positions, atom_types)):
            # Default size and color
            size = 50
            color = self.atom_colors.get(atom_type, 'gray')
            alpha = 0.8
            
            # Highlight specific atoms if requested
            if highlight is not None and i in highlight:
                size = 100
                alpha = 1.0
                
            ax.scatter(pos[0], pos[1], pos[2], color=color, s=size, alpha=alpha)
        
        # Plot bonds
        if bonds is not None:
            for a, b in bonds:
                if a < len(positions) and b < len(positions):
                    ax.plot([positions[a][0], positions[b][0]],
                           [positions[a][1], positions[b][1]],
                           [positions[a][2], positions[b][2]], 'k-', alpha=0.3, linewidth=1)
        
        # Mark damage sites
        if damage_sites is not None:
            for a, b in damage_sites:
                if b == -1:  # Single atom damage
                    if a < len(positions):
                        ax.scatter(positions[a][0], positions[a][1], positions[a][2], 
                                 color='black', s=100, marker='x')
                else:  # Bond damage
                    if a < len(positions) and b < len(positions):
                        ax.plot([positions[a][0], positions[b][0]],
                               [positions[a][1], positions[b][1]],
                               [positions[a][2], positions[b][2]], 'r--', linewidth=2)
        
        # Add legend for atom types
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, 
                      label=f'Type {t}', markersize=10)
            for t, color in self.atom_colors.items() if t in atom_types
        ]
        
        if damage_sites:
            legend_elements.append(
                plt.Line2D([0], [0], color='r', linestyle='--', label='Damage Site')
            )
            
        ax.legend(handles=legend_elements, loc='upper right')
        
        # Set axis labels and title
        ax.set_xlabel('X (nm)')
        ax.set_ylabel('Y (nm)')
        ax.set_zlabel('Z (nm)')
        ax.set_title('Material Visualization')
        
        # Set axis limits
        box_size = self.config.box_size
        ax.set_xlim(0, box_size[0])
        ax.set_ylim(0, box_size[1])
        ax.set_zlim(0, box_size[2])
        
        # Save or show the figure
        if filename:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.tight_layout()
            plt.show()
    
    def create_animation(self, trajectories, atom_types, bonds_list, damage_sites=None, filename="healing_animation.gif", fps=10):
        """Create an animation of the healing process"""
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Set axis limits
        box_size = self.config.box_size
        ax.set_xlim(0, box_size[0])
        ax.set_ylim(0, box_size[1])
        ax.set_zlim(0, box_size[2])
        
        # Set labels
        ax.set_xlabel('X (nm)')
        ax.set_ylabel('Y (nm)')
        ax.set_zlabel('Z (nm)')
        ax.set_title('Self-Healing Process')
        
        # Function to update the plot for each frame
        scatter_plots = []
        line_plots = []
        
        # Initialize with first frame
        for atom_type in range(10):
            mask = atom_types == atom_type
            if np.any(mask):
                color = self.atom_colors.get(atom_type, 'gray')
                scatter = ax.scatter([], [], [], color=color, s=50, alpha=0.8)
                scatter_plots.append((scatter, atom_type))
        
        # Initialize animation function
        def animate(i):
            # Clear previous frame's bonds
            for line in line_plots:
                line.remove()
            line_plots.clear()
            
            # Update atom positions
            for scatter, atom_type in scatter_plots:
                mask = atom_types == atom_type
                if np.any(mask):
                    scatter._offsets3d = (
                        trajectories[i][mask, 0],
                        trajectories[i][mask, 1],
                        trajectories[i][mask, 2]
                    )
            
            # Update bonds
            if i < len(bonds_list):
                for a, b in bonds_list[i]:
                    if a < len(trajectories[i]) and b < len(trajectories[i]):
                        line, = ax.plot(
                            [trajectories[i][a][0], trajectories[i][b][0]],
                            [trajectories[i][a][1], trajectories[i][b][1]],
                            [trajectories[i][a][2], trajectories[i][b][2]],
                            'k-', alpha=0.3, linewidth=1
                        )
                        line_plots.append(line)
            
            # Update damage sites if available
            if damage_sites:
                for a, b in damage_sites:
                    if b == -1 and a < len(trajectories[i]):  # Single atom damage
                        line, = ax.plot(
                            [trajectories[i][a][0]], 
                            [trajectories[i][a][1]], 
                            [trajectories[i][a][2]],
                            'rx', markersize=10
                        )
                        line_plots.append(line)
                    elif a < len(trajectories[i]) and b < len(trajectories[i]):  # Bond damage
                        line, = ax.plot(
                            [trajectories[i][a][0], trajectories[i][b][0]],
                            [trajectories[i][a][1], trajectories[i][b][1]],
                            [trajectories[i][a][2], trajectories[i][b][2]],
                            'r--', linewidth=2
                        )
                        line_plots.append(line)
            
            return scatter_plots + line_plots
        
        # Create animation
        ani = FuncAnimation(
            fig, animate, frames=len(trajectories),
            interval=1000/fps, blit=False
        )
        
        # Save as GIF
        ani.save(filename, writer='pillow', fps=fps, dpi=100)
        plt.close()
        
        logger.info(f"Animation saved as {filename}")
    
    def plot_healing_progress(self, integrity_over_time, damaged_integrity, initial_integrity, energy_over_time=None, filename=None):
        """Plot the healing progress over time"""
        plt.figure(figsize=(14, 8))
        
        # Plot material integrity
        plt.subplot(1, 2, 1)
        plt.plot(integrity_over_time, 'b-', linewidth=2)
        plt.axhline(y=damaged_integrity, color='r', linestyle='--', label='Damaged')
        plt.axhline(y=initial_integrity, color='g', linestyle='--', label='Initial')
        
        plt.xlabel('Simulation Steps')
        plt.ylabel('Material Integrity (%)')
        plt.title('Self-Healing Progress')
        plt.legend()
        plt.grid(True)
        
        # Plot energy if available
        if energy_over_time:
            plt.subplot(1, 2, 2)
            for energy_type, values in energy_over_time.items():
                if energy_type in ['potential', 'kinetic', 'total']:
                    plt.plot(values, label=energy_type.capitalize())
            
            plt.xlabel('Simulation Steps')
            plt.ylabel('Energy')
            plt.title('Energy Evolution')
            plt.legend()
            plt.grid(True)
        
        plt.tight_layout()
        
        # Save or show the figure
        if filename:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def visualize_bond_network(self, positions, bonds, atom_types, filename=None):
        """Visualize the bond network as a 2D graph"""
        plt.figure(figsize=(12, 10))
        
        # Create a graph layout
        G = {}
        for a, b in bonds:
            if a not in G:
                G[a] = []
            if b not in G:
                G[b] = []
            
            G[a].append(b)
            G[b].append(a)
        
        # Use t-SNE to project positions to 2D
        tsne = TSNE(n_components=2, perplexity=min(30, len(positions)-1))
        positions_2d = tsne.fit_transform(positions)
        
        # Plot nodes and edges
        for i, pos in enumerate(positions_2d):
            color = self.atom_colors.get(atom_types[i], 'gray')
            plt.scatter(pos[0], pos[1], color=color, s=100, alpha=0.8)
            
            # Plot bonds
            neighbors = G.get(i, [])
            for j in neighbors:
                if j > i:  # Only plot each bond once
                    plt.plot(
                        [positions_2d[i][0], positions_2d[j][0]],
                        [positions_2d[i][1], positions_2d[j][1]],
                        'k-', alpha=0.3
                    )
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, 
                      label=f'Type {t}', markersize=10)
            for t, color in self.atom_colors.items() if t in atom_types
        ]
        plt.legend(handles=legend_elements, loc='upper right')
        
        plt.title('Bond Network Graph')
        plt.axis('off')
        
        # Save or show
        if filename:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.tight_layout()
            plt.show()


def main():
    """Main function to run the simulation and training"""
    # Create configuration
    config = SimulationConfig()
    
    # Save the configuration
    config.save(os.path.join(config.output_dir, "simulation_config.json"))
    
    logger.info(f"Starting simulation with device: {config.device}")
    
    # Initialize simulator for different materials
    md_sim = MDSimulator(config)
    
    # Run simulations for different material types
    material_types = ["polymer", "capsule", "hydrogel", "nanocomposite"]
    simulation_results = {}
    
    for material_type in material_types:
        logger.info(f"Simulating {material_type} material")
        
        # Initialize material
        md_sim = MDSimulator(config)
        md_sim.initialize_system(material_type)
        
        # Run initial simulation to equilibrate
        undamaged_results = md_sim.run_simulation(n_steps=1000)
        
        # Apply damage
        md_sim.apply_damage(damage_type="crack", intensity=0.7)
        
        # Simulate damaged material
        damaged_results = md_sim.run_simulation(n_steps=1000)
        
        # Store results
        simulation_results[material_type] = {
            "undamaged": undamaged_results,
            "damaged": damaged_results
        }
    
    # Create combined dataset for model training
    logger.info("Creating training dataset")
    
    all_trajectories = []
    all_velocities = []
    all_bonds = []
    all_atom_types = []
    
    for material_type, results in simulation_results.items():
        # Combine trajectories
        all_trajectories.append(results["undamaged"]["trajectories"])
        all_trajectories.append(results["damaged"]["trajectories"])
        
        # Combine velocities
        all_velocities.append(results["undamaged"]["velocities"])
        all_velocities.append(results["damaged"]["velocities"])
        
        # Combine bonds
        all_bonds.extend(results["undamaged"]["bond_data"])
        all_bonds.extend(results["damaged"]["bond_data"])
        
        # Store atom types (use the last one as they should be consistent)
        all_atom_types.append(results["damaged"]["final_state"]["atom_types"])
    
    # Concatenate trajectories and velocities
    combined_trajectories = np.concatenate(all_trajectories)
    combined_velocities = np.concatenate(all_velocities)
    
    # Use the atom types from the most complex material
    # For simplicity, we'll just use the last one
    combined_atom_types = all_atom_types[-1]
    
    # Create the dataset
    dataset = MDTrajectoryDataset(
        combined_trajectories,
        combined_velocities,
        combined_atom_types,
        all_bonds,
        md_sim.damage_sites
    )
    
    # Split into train and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True,
        num_workers=min(4, config.n_cores)
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.batch_size,
        num_workers=min(4, config.n_cores)
    )
    
    # Initialize model based on chosen type
    logger.info(f"Initializing {config.model_type} model")
    
    if config.model_type == "transformer":
        model = TransformerHealingAgent(config).to(config.device)
    elif config.model_type == "gcn":
        model = GCNHealingAgent(config).to(config.device)
    elif config.model_type == "hybrid":
        model = HybridHealingAgent(config).to(config.device)
    else:
        raise ValueError(f"Unknown model type: {config.model_type}")
    
    # Train the model
    logger.info("Starting model training")
    training_results = train_healing_agent(model, train_loader, val_loader, config)
    
    # Load the best model for evaluation
    best_model_path = os.path.join(config.model_save_path, f"best_{config.model_type}_agent.pt")
    model.load_state_dict(torch.load(best_model_path))
    
    # Run self-healing simulation with the trained model
    logger.info("Simulating self-healing with trained model")
    
    # Test healing on polymer material
    md_sim = MDSimulator(config)
    md_sim.initialize_system("polymer")
    
    # Apply damage
    md_sim.apply_damage(damage_type="crack", intensity=0.7)
    
    # Simulate healing
    healed_results = md_sim.run_simulation(n_steps=5000, healing_agent=model)
    
    # Create visualizer and analyzer
    visualizer = Visualizer(config)
    analyzer = MaterialAnalyzer(config)
    
    # Visualize results
    logger.info("Generating visualizations")
    
    # Original, damaged, and healed states
    original_positions = simulation_results["polymer"]["undamaged"]["final_state"]["positions"]
    original_atom_types = simulation_results["polymer"]["undamaged"]["final_state"]["atom_types"]
    original_bonds = simulation_results["polymer"]["undamaged"]["final_state"]["bonds"]
    
    damaged_positions = simulation_results["polymer"]["damaged"]["final_state"]["positions"]
    damaged_atom_types = simulation_results["polymer"]["damaged"]["final_state"]["atom_types"]
    damaged_bonds = simulation_results["polymer"]["damaged"]["final_state"]["bonds"]
    
    healed_positions = healed_results["final_state"]["positions"]
    healed_atom_types = healed_results["final_state"]["atom_types"]
    healed_bonds = healed_results["final_state"]["bonds"]
    
    # Create visualization directories
    vis_dir = os.path.join(config.output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Visualize states
    visualizer.visualize_material(
        original_positions,
        original_atom_types,
        original_bonds,
        filename=os.path.join(vis_dir, "original_material.png")
    )
    
    visualizer.visualize_material(
        damaged_positions,
        damaged_atom_types,
        damaged_bonds,
        damage_sites=md_sim.damage_sites,
        filename=os.path.join(vis_dir, "damaged_material.png")
    )
    
    visualizer.visualize_material(
        healed_positions,
        healed_atom_types,
        healed_bonds,
        filename=os.path.join(vis_dir, "healed_material.png")
    )
    
    # Create animation of healing process
    visualizer.create_animation(
        healed_results["trajectories"],
        healed_atom_types,
        healed_results["bond_data"],
        md_sim.damage_sites,
        filename=os.path.join(vis_dir, "healing_animation.gif")
    )
    
    # Visualize bond network
    visualizer.visualize_bond_network(
        original_positions,
        original_bonds,
        original_atom_types,
        filename=os.path.join(vis_dir, "original_network.png")
    )
    
    visualizer.visualize_bond_network(
        damaged_positions,
        damaged_bonds,
        damaged_atom_types,
        filename=os.path.join(vis_dir, "damaged_network.png")
    )
    
    visualizer.visualize_bond_network(
        healed_positions,
        healed_bonds,
        healed_atom_types,
        filename=os.path.join(vis_dir, "healed_network.png")
    )
    
    # Analyze material integrity
    logger.info("Analyzing material properties")
    
    # Calculate integrity over time
    integrity_over_time = []
    for bonds in healed_results["bond_data"]:
        integrity = analyzer.analyze_material_integrity(bonds, original_bonds)
        integrity_over_time.append(integrity)
    
    # Calculate initial and damaged integrity
    initial_integrity = 100.0
    damaged_integrity = analyzer.analyze_material_integrity(damaged_bonds, original_bonds)
    healed_integrity = analyzer.analyze_material_integrity(healed_bonds, original_bonds)
    
    # Extract energy data
    energy_over_time = {
        'potential': [data['potential'] for data in healed_results["energy_data"]],
        'kinetic': [data['kinetic'] for data in healed_results["energy_data"]],
        'total': [data['total'] for data in healed_results["energy_data"]]
    }
    
    # Plot healing progress
    visualizer.plot_healing_progress(
        integrity_over_time,
        damaged_integrity,
        initial_integrity,
        energy_over_time,
        filename=os.path.join(vis_dir, "healing_progress.png")
    )
    
    # Calculate healing efficiency
    healing_metrics = analyzer.analyze_healing_efficiency(
        original_bonds, damaged_bonds, healed_bonds
    )
    
    # Calculate structural recovery
    structure_metrics = analyzer.analyze_structural_recovery(
        original_positions, damaged_positions, healed_positions
    )
    
    # Calculate mechanical properties
    mechanical_metrics_original = analyzer.analyze_mechanical_properties(
        original_positions, original_bonds, original_atom_types
    )
    
    mechanical_metrics_damaged = analyzer.analyze_mechanical_properties(
        damaged_positions, damaged_bonds, damaged_atom_types
    )
    
    mechanical_metrics_healed = analyzer.analyze_mechanical_properties(
        healed_positions, healed_bonds, healed_atom_types
    )
    
    # Compile all results
    final_results = {
        "training": training_results,
        "simulation": {
            "polymer": simulation_results["polymer"],
            "healed": healed_results
        },
        "analysis": {
            "healing_metrics": healing_metrics,
            "structure_metrics": structure_metrics,
            "mechanical": {
                "original": mechanical_metrics_original,
                "damaged": mechanical_metrics_damaged,
                "healed": mechanical_metrics_healed
            },
            "integrity": {
                "initial": initial_integrity,
                "damaged": damaged_integrity,
                "healed": healed_integrity,
                "over_time": integrity_over_time
            }
        }
    }
    
    # Save results
    logger.info("Saving final results")
    
    results_file = os.path.join(config.output_dir, "healing_results.json")
    with open(results_file, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(i) for i in obj]
            else:
                return obj
                
        # Convert numpy values in the results
        serializable_results = convert_numpy(final_results)
        
        # Filter out non-serializable data
        filtered_results = {
            "training": {
                "train_losses": serializable_results["training"]["train_losses"],
                "val_losses": serializable_results["training"]["val_losses"],
                "best_val_loss": serializable_results["training"]["best_val_loss"]
            },
            "analysis": serializable_results["analysis"]
        }
        
        json.dump(filtered_results, f, indent=4)
    
    # Print summary
    logger.info("\nSelf-Healing Material Simulation Summary:")
    logger.info(f"Material Integrity: Original: 100.0%, Damaged: {damaged_integrity:.2f}%, Healed: {healed_integrity:.2f}%")
    logger.info(f"Healing Efficiency: {healing_metrics['healing_efficiency']:.2f}%")
    logger.info(f"Structural Recovery: {structure_metrics['structural_recovery']:.2f}%")
    logger.info(f"Mechanical Properties Recovery: {mechanical_metrics_healed['estimated_youngs_modulus']/mechanical_metrics_original['estimated_youngs_modulus']*100:.2f}%")
    
    # Save full trajectory data separately (using pickle)
    trajectory_file = os.path.join(config.output_dir, "trajectory_data.pkl")
    with open(trajectory_file, 'wb') as f:
        pickle.dump({
            "undamaged": simulation_results["polymer"]["undamaged"]["trajectories"],
            "damaged": simulation_results["polymer"]["damaged"]["trajectories"],
            "healed": healed_results["trajectories"]
        }, f)
    
    logger.info(f"Results saved to {config.output_dir}")

# Run the main function
if __name__ == "__main__":
    main()
