import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
import pickle
import time
from tqdm import tqdm

# Konfiguration
CONFIG = {
    "n_atoms": 1000,              # Anzahl der Atome im System
    "box_size": [10.0, 10.0, 10.0], # Simulationsbox-Größe in nm
    "time_step": 0.002,           # MD-Zeitschritt in ps
    "n_steps": 10000,             # Anzahl der Simulationsschritte
    "temperature": 300,           # Temperatur in K
    "model_dim": 256,             # Dimension des Transformer-Modells
    "n_heads": 8,                 # Anzahl der Attention-Heads
    "n_layers": 6,                # Anzahl der Transformer-Layers
    "dropout": 0.1,               # Dropout-Rate
    "lr": 1e-4,                   # Lernrate
    "batch_size": 16,             # Batch-Größe
    "epochs": 50,                 # Anzahl der Trainingsepochen
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}

print(f"Gerät für Berechnungen: {CONFIG['device']}")

# Molekulare Dynamik-Simulator
class MDSimulator:
    def __init__(self, config):
        self.config = config
        self.positions = np.zeros((config["n_atoms"], 3))
        self.velocities = np.zeros((config["n_atoms"], 3))
        self.forces = np.zeros((config["n_atoms"], 3))
        self.atom_types = np.zeros(config["n_atoms"], dtype=int)
        self.bonds = []
        self.damage_sites = []
        
    def initialize_system(self, material_type="polymer"):
        """Initialisiert ein System basierend auf dem Materialtyp"""
        # Zufällige Positionen in der Box
        box = self.config["box_size"]
        self.positions = np.random.rand(self.config["n_atoms"], 3) * box
        
        # Maxwell-Boltzmann-Verteilung für Geschwindigkeiten
        sigma = np.sqrt(self.config["temperature"] * 0.00831 / 1.0)  # in nm/ps
        self.velocities = np.random.normal(0, sigma, (self.config["n_atoms"], 3))
        
        # Verschiedene Materialtypen
        if material_type == "polymer":
            # Erstelle lineare Polymerketten
            chain_length = 20
            n_chains = self.config["n_atoms"] // chain_length
            
            for c in range(n_chains):
                start_idx = c * chain_length
                end_idx = (c + 1) * chain_length
                # Kettenatome markieren (Typ 1)
                self.atom_types[start_idx:end_idx] = 1
                
                # Kovalente Bindungen zwischen benachbarten Atomen
                for i in range(start_idx, end_idx - 1):
                    self.bonds.append((i, i + 1))
                    
            # Füge reversible Vernetzungspunkte hinzu (Typ 2)
            crosslink_sites = np.random.choice(
                np.where(self.atom_types == 1)[0], 
                size=self.config["n_atoms"] // 10, 
                replace=False
            )
            self.atom_types[crosslink_sites] = 2
        
        elif material_type == "capsule":
            # Erstelle Verbundwerkstoff mit Mikrokapseln
            # Matrix (Typ 1)
            self.atom_types[:] = 1
            
            # Mikrokapseln (Typ 3) und Heilungsagens (Typ 4)
            n_capsules = 5
            capsule_radius = 1.0
            capacity = 20
            
            for i in range(n_capsules):
                center = np.random.rand(3) * (np.array(box) - 2*capsule_radius) + capsule_radius
                
                # Finde Atome innerhalb der Kapsel
                distances = np.linalg.norm(self.positions - center, axis=1)
                capsule_atoms = np.where(distances < capsule_radius)[0]
                
                # Kapselwand
                shell_atoms = capsule_atoms[:len(capsule_atoms)//2]
                self.atom_types[shell_atoms] = 3
                
                # Heilungsagens innerhalb der Kapsel
                core_atoms = capsule_atoms[len(capsule_atoms)//2:]
                self.atom_types[core_atoms] = 4
    
    def apply_damage(self, damage_type="crack", intensity=0.5):
        """Wendet Schaden auf das Material an"""
        if damage_type == "crack":
            # Simuliere einen Riss durch Löschen von Bindungen
            n_bonds_to_break = int(len(self.bonds) * intensity * 0.1)
            bonds_to_break = np.random.choice(len(self.bonds), n_bonds_to_break, replace=False)
            
            for idx in bonds_to_break:
                a, b = self.bonds[idx]
                self.damage_sites.append((a, b))
                
            # Entferne die gebrochenen Bindungen
            self.bonds = [bond for i, bond in enumerate(self.bonds) if i not in bonds_to_break]
            
        elif damage_type == "impact":
            # Simuliere eine Aufprallstelle
            impact_center = np.random.rand(3) * self.config["box_size"]
            impact_radius = intensity * 2.0
            
            distances = np.linalg.norm(self.positions - impact_center, axis=1)
            affected_atoms = np.where(distances < impact_radius)[0]
            
            # Verschiebe die betroffenen Atome
            displacement = np.random.normal(0, 0.5, (len(affected_atoms), 3))
            self.positions[affected_atoms] += displacement
            
            self.damage_sites = [(a, -1) for a in affected_atoms]  # -1 bedeutet kein spezifischer Partner
    
    def calculate_forces(self):
        """Berechnet die Kräfte zwischen den Atomen"""
        # Reset forces
        self.forces.fill(0.0)
        
        # Lennard-Jones-Potential für nicht-gebundene Interaktionen
        for i in range(self.config["n_atoms"]):
            for j in range(i+1, self.config["n_atoms"]):
                # Prüfe, ob die Atome gebunden sind
                is_bonded = any((i, j) in self.bonds or (j, i) in self.bonds for bond in self.bonds)
                
                if not is_bonded:
                    r_ij = self.positions[j] - self.positions[i]
                    r = np.linalg.norm(r_ij)
                    
                    # Cutoff
                    if r < 2.5:
                        # Lennard-Jones-Parameter basierend auf Atomtypen anpassen
                        sigma = 0.3
                        epsilon = 0.1
                        
                        # Lennard-Jones
                        sr6 = (sigma/r)**6
                        f_ij = 24 * epsilon * (2 * sr6**2 - sr6) / r**2 * r_ij
                        
                        self.forces[i] += f_ij
                        self.forces[j] -= f_ij
        
        # Harmonisches Potential für kovalente Bindungen
        for a, b in self.bonds:
            r_ij = self.positions[b] - self.positions[a]
            r = np.linalg.norm(r_ij)
            
            # Gleichgewichtsabstand und Kraftkonstante
            r0 = 0.15  # nm
            k = 500.0  # kJ/(mol*nm^2)
            
            # Harmonische Kraft
            f_ij = k * (r - r0) * r_ij / r
            
            self.forces[a] += f_ij
            self.forces[b] -= f_ij
            
        # Spezielle Kräfte für reversible Vernetzungspunkte (Typ 2)
        crosslink_sites = np.where(self.atom_types == 2)[0]
        for i in crosslink_sites:
            for j in crosslink_sites:
                if i != j:
                    r_ij = self.positions[j] - self.positions[i]
                    r = np.linalg.norm(r_ij)
                    
                    # Anziehende Kraft innerhalb einer bestimmten Entfernung
                    if 0.3 < r < 0.8:
                        f_ij = 2.0 * np.exp(-(r-0.5)**2/0.1) * r_ij / r
                        self.forces[i] += f_ij
                        self.forces[j] -= f_ij
    
    def step(self):
        """Führt einen Zeitschritt in der MD-Simulation durch"""
        dt = self.config["time_step"]
        
        # Berechne Kräfte
        self.calculate_forces()
        
        # Velocity Verlet-Integration
        self.positions += self.velocities * dt + 0.5 * self.forces * dt**2
        self.velocities += 0.5 * self.forces * dt
        
        # Periodische Randbedingungen
        self.positions = np.mod(self.positions, self.config["box_size"])
        
        # Berechne Kräfte erneut mit den aktualisierten Positionen
        old_forces = self.forces.copy()
        self.calculate_forces()
        
        # Abschließender Velocity-Update
        self.velocities += 0.5 * self.forces * dt
        
        # Temperaturkontrolle durch Berendsen-Thermostat
        tau = 0.1  # ps
        T_current = np.sum(self.velocities**2)
        scaling_factor = np.sqrt(1 + (dt/tau) * (self.config["temperature"]/T_current - 1))
        self.velocities *= scaling_factor
    
    def run_simulation(self, n_steps=None, healing_agent=None):
        """Führt die MD-Simulation für n_steps durch"""
        if n_steps is None:
            n_steps = self.config["n_steps"]
            
        trajectories = []
        bond_data = []
        
        for step in tqdm(range(n_steps), desc="MD-Simulation"):
            # Speichere den aktuellen Zustand
            trajectories.append(self.positions.copy())
            bond_data.append(self.bonds.copy())
            
            # Führe einen Zeitschritt durch
            self.step()
            
            # Wende den KI-Heilungsagenten an (falls vorhanden)
            if healing_agent is not None and step > n_steps // 3:
                if step % 100 == 0:  # Nicht in jedem Schritt anwenden
                    self.apply_healing_agent(healing_agent)
        
        return {
            "trajectories": np.array(trajectories),
            "bond_data": bond_data,
            "final_state": {
                "positions": self.positions.copy(),
                "velocities": self.velocities.copy(),
                "atom_types": self.atom_types.copy(),
                "bonds": self.bonds.copy()
            }
        }
    
    def apply_healing_agent(self, healing_agent):
        """Wendet den KI-Heilungsagenten an, um Schäden zu reparieren"""
        # Konvertiere den Zustand für das Transformer-Modell
        state_tensor = self.get_state_tensor()
        
        # Übergebe den Zustand an den Heilungsagenten
        healing_actions = healing_agent.predict_healing_actions(state_tensor)
        
        # Wende die vorhergesagten Heilungsaktionen an
        for action in healing_actions:
            action_type, params = action
            
            if action_type == "form_bond":
                a, b = params
                if 0 <= a < self.config["n_atoms"] and 0 <= b < self.config["n_atoms"]:
                    # Prüfe, ob die Atome nahe genug sind
                    r_ab = np.linalg.norm(self.positions[a] - self.positions[b])
                    if r_ab < 0.3:  # Maximale Bindungsdistanz
                        self.bonds.append((a, b))
                        
            elif action_type == "activate_healing_agent":
                center, radius = params
                # Finde Atome vom Typ 4 (Heilungsagens) in der Nähe
                distances = np.linalg.norm(self.positions - center, axis=1)
                healing_atoms = np.where((distances < radius) & (self.atom_types == 4))[0]
                
                # Ändere ihren Typ auf 5 (aktiviertes Heilungsagens)
                self.atom_types[healing_atoms] = 5
    
    def get_state_tensor(self):
        """Konvertiert den aktuellen Zustand in einen Tensor für das Transformer-Modell"""
        # Positions-Embedding
        pos_normalized = self.positions / np.array(self.config["box_size"])
        
        # One-hot encoding für Atomtypen
        max_type = max(5, np.max(self.atom_types) + 1)
        type_onehot = np.zeros((self.config["n_atoms"], max_type))
        for i in range(self.config["n_atoms"]):
            type_onehot[i, self.atom_types[i]] = 1
            
        # Bindungsinformationen als Adjazenzmatrix
        bond_matrix = np.zeros((self.config["n_atoms"], self.config["n_atoms"]))
        for a, b in self.bonds:
            bond_matrix[a, b] = 1
            bond_matrix[b, a] = 1
            
        # Kombiniere die Informationen
        state = np.concatenate([
            pos_normalized,
            type_onehot,
        ], axis=1)
        
        return torch.FloatTensor(state)

# Dataset für das Training des Transformer-Modells
class MDTrajectoryDataset(Dataset):
    def __init__(self, trajectories, atom_types, bonds):
        self.trajectories = trajectories
        self.atom_types = atom_types
        self.bonds = bonds
        
    def __len__(self):
        return len(self.trajectories) - 1
    
    def __getitem__(self, idx):
        # Aktueller und nächster Zustand
        current_state = self.trajectories[idx]
        next_state = self.trajectories[idx + 1]
        
        # Konvertiere in Tensoren
        current = torch.FloatTensor(current_state)
        target = torch.FloatTensor(next_state)
        
        return {
            "current": current,
            "target": target,
            "atom_types": torch.LongTensor(self.atom_types),
            "bonds": self.bonds[idx]
        }

# Positional Encoding für den Transformer
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# Transformer-basierter Heilungsagent
class TransformerHealingAgent(nn.Module):
    def __init__(self, config):
        super(TransformerHealingAgent, self).__init__()
        self.config = config
        
        # Embedding-Dimensionen
        d_model = config["model_dim"]
        
        # Input-Projektionen
        self.position_embedding = nn.Linear(3, d_model // 4)
        self.type_embedding = nn.Linear(6, d_model // 4)  # 6 Atomtypen
        self.feature_proj = nn.Linear(d_model // 2, d_model)
        
        # Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer-Encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=config["n_heads"],
            dim_feedforward=d_model * 4,
            dropout=config["dropout"],
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, 
            num_layers=config["n_layers"]
        )
        
        # Ausgabe-Layer
        self.bond_predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
        
        self.position_predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 3)
        )
        
    def forward(self, x, mask=None):
        # x hat die Form [batch_size, n_atoms, features]
        batch_size, n_atoms, _ = x.shape
        
        # Extrahiere Positionsdaten und Atomtypen
        positions = x[:, :, :3]
        atom_types = x[:, :, 3:9]
        
        # Embeddings
        pos_emb = self.position_embedding(positions)
        type_emb = self.type_embedding(atom_types)
        
        # Kombiniere Embeddings
        combined = torch.cat([pos_emb, type_emb], dim=2)
        x = self.feature_proj(combined)
        
        # Positional Encoding
        x = self.pos_encoder(x)
        
        # Transformer-Encoder
        if mask is None:
            x = self.transformer_encoder(x)
        else:
            x = self.transformer_encoder(x, src_key_padding_mask=mask)
        
        # Vorhersagen
        bond_logits = self.bond_predictor(x)
        pos_delta = self.position_predictor(x)
        
        return bond_logits, pos_delta
    
    def predict_healing_actions(self, state_tensor):
        """Sagt Heilungsaktionen basierend auf dem aktuellen Zustand voraus"""
        self.eval()
        with torch.no_grad():
            x = state_tensor.unsqueeze(0).to(self.config["device"])
            bond_probs, pos_delta = self.forward(x)
            
            # Wandle Vorhersagen in konkrete Aktionen um
            bond_probs = bond_probs.squeeze(0).squeeze(-1).cpu().numpy()
            pos_delta = pos_delta.squeeze(0).cpu().numpy()
            
            # Finde potenzielle neue Bindungen
            actions = []
            
            # Betrachte nur Paare von Atomen, die aktuell geschädigt sind
            n_atoms = bond_probs.shape[0]
            for i in range(n_atoms):
                for j in range(i+1, n_atoms):
                    if bond_probs[i, j] > 0.8:  # Schwellenwert für Bindungsbildung
                        actions.append(("form_bond", (i, j)))
            
            # Aktiviere Heilungsagenten nahe geschädigter Bereiche
            damage_centers = []
            for i in range(n_atoms):
                # Wenn das Atom eine große vorhergesagte Positionsänderung hat,
                # könnte es auf Schäden hinweisen
                delta_mag = np.linalg.norm(pos_delta[i])
                if delta_mag > 0.1:
                    damage_centers.append(("activate_healing_agent", 
                                         (state_tensor[i, :3].numpy(), 0.5)))
            
            # Beschränke auf maximal 5 Heilungsaktionen pro Schritt
            if len(actions) > 5:
                actions = actions[:5]
                
            return actions

# Trainingsschleife für den Transformer
def train_healing_agent(model, train_loader, val_loader, config):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    
    # Verlustfunktionen
    bond_criterion = nn.BCELoss()
    pos_criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(config["epochs"]):
        model.train()
        epoch_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}"):
            optimizer.zero_grad()
            
            # Daten auf das richtige Gerät verschieben
            current = batch["current"].to(config["device"])
            target = batch["target"].to(config["device"])
            atom_types = batch["atom_types"].to(config["device"])
            
            # Vorwärtsdurchlauf
            bond_probs, pos_delta = model(current)
            
            # Bond-Labels erstellen
            bond_labels = torch.zeros_like(bond_probs)
            for i, bonds in enumerate(batch["bonds"]):
                for a, b in bonds:
                    if a < bond_labels.shape[1] and b < bond_labels.shape[1]:
                        bond_labels[i, a, b] = 1.0
                        bond_labels[i, b, a] = 1.0
            
            # Positionsdifferenz
            pos_targets = target[:, :, :3] - current[:, :, :3]
            
            # Verlust berechnen
            bond_loss = bond_criterion(bond_probs, bond_labels)
            pos_loss = pos_criterion(pos_delta, pos_targets)
            
            loss = bond_loss + pos_loss
            
            # Backpropagation
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validierung
        val_loss = validate_model(model, val_loader, config)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {val_loss:.4f}")
        
        # Modell speichern, wenn es besser ist
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_healing_agent.pt")
    
    return train_losses, val_losses

def validate_model(model, val_loader, config):
    model.eval()
    bond_criterion = nn.BCELoss()
    pos_criterion = nn.MSELoss()
    
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in val_loader:
            # Daten auf das richtige Gerät verschieben
            current = batch["current"].to(config["device"])
            target = batch["target"].to(config["device"])
            atom_types = batch["atom_types"].to(config["device"])
            
            # Vorwärtsdurchlauf
            bond_probs, pos_delta = model(current)
            
            # Bond-Labels erstellen
            bond_labels = torch.zeros_like(bond_probs)
            for i, bonds in enumerate(batch["bonds"]):
                for a, b in bonds:
                    if a < bond_labels.shape[1] and b < bond_labels.shape[1]:
                        bond_labels[i, a, b] = 1.0
                        bond_labels[i, b, a] = 1.0
            
            # Positionsdifferenz
            pos_targets = target[:, :, :3] - current[:, :, :3]
            
            # Verlust berechnen
            bond_loss = bond_criterion(bond_probs, bond_labels)
            pos_loss = pos_criterion(pos_delta, pos_targets)
            
            loss = bond_loss + pos_loss
            total_loss += loss.item()
    
    return total_loss / len(val_loader)

# Visualisierungsfunktionen
def visualize_material(positions, atom_types, bonds=None, damage_sites=None, filename=None):
    """Visualisiert den Materialzustand"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Farbzuordnung für Atomtypen
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'cyan']
    
    # Zeichne Atome
    for i, (pos, atom_type) in enumerate(zip(positions, atom_types)):
        ax.scatter(pos[0], pos[1], pos[2], color=colors[atom_type], s=50)
    
    # Zeichne Bindungen
    if bonds is not None:
        for a, b in bonds:
            if a < len(positions) and b < len(positions):
                ax.plot([positions[a][0], positions[b][0]],
                        [positions[a][1], positions[b][1]],
                        [positions[a][2], positions[b][2]], 'k-', alpha=0.3)
    
    # Markiere Schadensstellen
    if damage_sites is not None:
        for a, b in damage_sites:
            if b == -1:  # Einzelnes Atom
                if a < len(positions):
                    ax.scatter(positions[a][0], positions[a][1], positions[a][2], 
                              color='black', s=100, marker='x')
            else:  # Bindung
                if a < len(positions) and b < len(positions):
                    ax.plot([positions[a][0], positions[b][0]],
                           [positions[a][1], positions[b][1]],
                           [positions[a][2], positions[b][2]], 'r--', linewidth=2)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Materialvisualisierung')
    
    if filename:
        plt.savefig(filename)
    else:
        plt.show()

def visualize_healing_process(trajectories, atom_types, bonds_list, damage_sites, output_dir="healing_process"):
    """Erstellt eine Reihe von Bildern, die den Heilungsprozess darstellen"""
    os.makedirs(output_dir, exist_ok=True)
    
    for i, (positions, bonds) in enumerate(zip(trajectories, bonds_list)):
        if i % 100 == 0:  # Speichere nur jeden 100. Frame
            filename = os.path.join(output_dir, f"frame_{i:04d}.png")
            visualize_material(positions, atom_types, bonds, damage_sites, filename)
    
    print(f"Visualisierungen in {output_dir} gespeichert")

def generate_animation(image_dir, output_file="healing_animation.gif"):
    """Erstellt eine Animation aus den gespeicherten Bildern"""
    try:
        import imageio
        
        images = []
        image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
        
        for file in image_files:
            images.append(imageio.imread(os.path.join(image_dir, file)))
        
        imageio.mimsave(output_file, images, duration=0.1)
        print(f"Animation gespeichert als {output_file}")
    
    except ImportError:
        print("Imageio ist erforderlich für Animationen. Installiere es mit 'pip install imageio'.")

# Hauptfunktion für die MD-Simulation und das Modelltraining
def main():
    # Simulationsumgebung initialisieren
    md_sim = MDSimulator(CONFIG)
    md_sim.initialize_system(material_type="polymer")
    
    # Simuliere unbeschädigtes Material
    print("Simulation des unbeschädigten Materials...")
    undamaged_results = md_sim.run_simulation(n_steps=2000)
    
    # Füge Schaden hinzu
    print("Füge Schaden zum Material hinzu...")
    md_sim.apply_damage(damage_type="crack", intensity=0.7)
    
    # Simuliere beschädigtes Material ohne Heilung
    print("Simulation des beschädigten Materials ohne Heilung...")
    damaged_results = md_sim.run_simulation(n_steps=2000)
    
    # Erstelle Trainingsdaten
    print("Erstelle Trainingsdaten aus den Simulationen...")
    train_trajectories = np.concatenate([
        undamaged_results["trajectories"],
        damaged_results["trajectories"]
    ])
    
    # Kombiniere Bindungsdaten
    train_bonds = undamaged_results["bond_data"] + damaged_results["bond_data"]
    
    # Erstelle Dataset
    dataset = MDTrajectoryDataset(
        train_trajectories,
        md_sim.atom_types,
        train_bonds
    )
    
    # Trainings- und Validierungsdaten aufteilen
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # DataLoader erstellen
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"])
    
    # Transformer-Modell initialisieren
    model = TransformerHealingAgent(CONFIG).to(CONFIG["device"])
    
    # Modell trainieren
    print("Trainiere das Transformer-Modell...")
    train_losses, val_losses = train_healing_agent(model, train_loader, val_loader, CONFIG)
    
    # Lade das beste Modell
    model.load_state_dict(torch.load("best_healing_agent.pt"))
    
    # Simuliere die Selbstheilung mit dem trainierten Modell
    print("Simulation der Selbstheilung mit dem KI-Modell...")
    md_sim = MDSimulator(CONFIG)
    md_sim.initialize_system(material_type="polymer")
    md_sim.apply_damage(damage_type="crack", intensity=0.7)
    
    healed_results = md_sim.run_simulation(n_steps=5000, healing_agent=model)
    
    # Visualisiere die Ergebnisse
    print("Visualisiere die Ergebnisse...")
    visualize_material(
        undamaged_results["final_state"]["positions"],
        undamaged_results["final_state"]["atom_types"],
        undamaged_results["final_state"]["bonds"],
        filename="undamaged_material.png"
    )
    
    visualize_material(
        damaged_results["final_state"]["positions"],
        damaged_results["final_state"]["atom_types"],
        damaged_results["final_state"]["bonds"],
        md_sim.damage_sites,
        filename="damaged_material.png"
    )
    
    visualize_material(
        healed_results["final_state"]["positions"],
        healed_results["final_state"]["atom_types"],
        healed_results["final_state"]["bonds"],
        filename="healed_material.png"
    )
    
    # Erstelle Animation des Heilungsprozesses
    visualize_healing_process(
        healed_results["trajectories"],
        healed_results["final_state"]["atom_types"],
        healed_results["bond_data"],
        md_sim.damage_sites
    )
    
    generate_animation("healing_process", "healing_animation.gif")
    
    # Analysiere die Ergebnisse
    def analyze_material_integrity(bonds, initial_bonds):
        """Berechnet die Materialintegrität basierend auf den Bindungen"""
        initial_count = len(initial_bonds)
        current_count = len(bonds)
        return (current_count / initial_count) * 100 if initial_count > 0 else 0
    
    initial_bonds = undamaged_results["final_state"]["bonds"]
    damaged_bonds = damaged_results["final_state"]["bonds"]
    healed_bonds = healed_results["final_state"]["bonds"]
    
    initial_integrity = 100.0
    damaged_integrity = analyze_material_integrity(damaged_bonds, initial_bonds)
    healed_integrity = analyze_material_integrity(healed_bonds, initial_bonds)
    
    print("\nMaterialintegritätsanalyse:")
    print(f"Unbeschädigtes Material: {initial_integrity:.2f}%")
    print(f"Beschädigtes Material: {damaged_integrity:.2f}%")
    print(f"Geheiltes Material: {healed_integrity:.2f}%")
    print(f"Wiederherstellungsrate: {(healed_integrity - damaged_integrity):.2f}%")
    
    # Plotte den Verlauf der Materialintegrität während der Heilung
    integrity_over_time = []
    for bonds in healed_results["bond_data"]:
        integrity = analyze_material_integrity(bonds, initial_bonds)
        integrity_over_time.append(integrity)
    
    plt.figure(figsize=(10, 6))
    plt.plot(integrity_over_time)
    plt.axhline(y=damaged_integrity, color='r', linestyle='--', label='Beschädigt')
    plt.axhline(y=initial_integrity, color='g', linestyle='--', label='Unbeschädigt')
    plt.xlabel('Simulationsschritte')
    plt.ylabel('Materialintegrität (%)')
    plt.title('Selbstheilungsprozess über Zeit')
    plt.legend()
    plt.grid(True)
    plt.savefig("healing_progress.png")
    
    # Analysiere die Bindungsenergie
    def calculate_bond_energy(positions, bonds):
        """Berechnet die Gesamtbindungsenergie"""
        energy = 0.0
        k = 500.0  # kJ/(mol*nm^2)
        r0 = 0.15  # nm
        
        for a, b in bonds:
            if a < len(positions) and b < len(positions):
                r_ij = positions[b] - positions[a]
                r = np.linalg.norm(r_ij)
                energy += 0.5 * k * (r - r0)**2
        
        return energy
    
    initial_energy = calculate_bond_energy(
        undamaged_results["final_state"]["positions"],
        initial_bonds
    )
    
    damaged_energy = calculate_bond_energy(
        damaged_results["final_state"]["positions"],
        damaged_bonds
    )
    
    healed_energy = calculate_bond_energy(
        healed_results["final_state"]["positions"],
        healed_bonds
    )
    
    print("\nBindungsenergieanalyse:")
    print(f"Unbeschädigtes Material: {initial_energy:.2f} kJ/mol")
    print(f"Beschädigtes Material: {damaged_energy:.2f} kJ/mol")
    print(f"Geheiltes Material: {healed_energy:.2f} kJ/mol")
    
    # Speichere die Ergebnisse
    results = {
        "undamaged": undamaged_results,
        "damaged": damaged_results,
        "healed": healed_results,
        "model": model.state_dict(),
        "config": CONFIG,
        "analysis": {
            "initial_integrity": initial_integrity,
            "damaged_integrity": damaged_integrity,
            "healed_integrity": healed_integrity,
            "initial_energy": initial_energy,
            "damaged_energy": damaged_energy,
            "healed_energy": healed_energy
        }
    }
    
    with open("self_healing_results.pkl", "wb") as f:
        pickle.dump(results, f)
    
    print("\nSimulation abgeschlossen und Ergebnisse gespeichert!")

# Führe das Hauptprogramm aus
if __name__ == "__main__":
    main()
