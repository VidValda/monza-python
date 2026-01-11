import numpy as np
import json
import csv
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.interpolate import interp1d
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

# --- Configuration ---
@dataclass
class PhysicsParams:
    dt: float = 0.01
    gravity: float = 9.81
    friction: float = 0.0223
    restitution: float = 0.1  # Bounciness
    sub_steps: int = 10       # Physics accuracy

@dataclass
class SimulationConfig:
    diff_path: str = 'dificultad2.json'
    circ_path: str = 'circulos.json'
    duration_steps: int = 2000
    max_tilt: float = 45.0    # Degrees
    max_omega: float = 8.0    # Rad/s
    setpoint: float = 0.0      # Target position for fuzzy controller (deprecated, using controller function)
    nivel: int = 1             # Current level (1-4, MATLAB 1-based)

# --- Helper Functions ---
def rotate_vector(x: float, y: float, angle: float) -> Tuple[float, float]:
    """Rotates a vector (x, y) by a given angle."""
    c, s = np.cos(angle), np.sin(angle)
    return x * c - y * s, x * s + y * c

def inverse_rotate_vector(x: float, y: float, angle: float) -> Tuple[float, float]:
    """Inverse rotation (global to local)."""
    c, s = np.cos(angle), np.sin(angle)
    return x * c + y * s, -x * s + y * c

class Level:
    def __init__(self, config: SimulationConfig):
        self.floors = {}
        self.visual_segments = []
        self._load_level_data(config.diff_path, config.circ_path)

    def _load_level_data(self, d_path, c_path):
        try:
            with open(d_path, 'r') as f: d_data = json.load(f)
            with open(c_path, 'r') as f: c_data = json.load(f)
        except FileNotFoundError:
            print(f"Error: Could not load level files.")
            return

        # Process Function segments (xp0, yp0, etc.)
        for i in range(20):
            xk = f'xp{i}' if i > 0 else 'xp'
            yk = f'yp{i}' if i > 0 else 'yp'
            
            if xk in d_data and yk in d_data:
                self._process_floor_segment(i, d_data[xk], d_data[yk])

        # Process Line segments
        i = 1
        while True:
            xk, yk = f'xl{i}', f'yl{i}'
            if xk not in d_data: break
            self.visual_segments.append((np.array(d_data[xk]).flatten(), np.array(d_data[yk]).flatten()))
            i += 1

        # Process Circle segments
        r_keys = sorted([k for k in c_data if k.startswith('r')], key=lambda x: int(x[1:]))
        for i in range(0, len(r_keys), 2):
            if i + 1 < len(r_keys):
                self.visual_segments.append((np.array(c_data[r_keys[i]]).flatten(), np.array(c_data[r_keys[i+1]]).flatten()))

    def _process_floor_segment(self, index, x_raw, y_raw):
        x = np.array(x_raw).flatten()
        y = np.array(y_raw).flatten()
        
        # Sort by X to ensure interpolation works
        idx = np.argsort(x)
        x, y = x[idx], y[idx]
        
        # Calculate slope derivatives
        dx = np.gradient(x)
        dx[dx == 0] = 1e-9 # Avoid div by zero
        slope = np.gradient(y) / dx

        self.floors[index] = {
            'func': interp1d(x, y, kind='cubic', fill_value="extrapolate"),
            'slope': interp1d(x, slope, kind='linear', fill_value="extrapolate"),
            'min': x[0],
            'max': x[-1]
        }
        self.visual_segments.append((x, y))

class PhysicsBall:
    def __init__(self, level: Level, start_idx=0):
        self.lvl = level
        self.params = PhysicsParams()
        
        # State
        self.current_floor_idx = start_idx
        self.previous_floor_idx = start_idx  # Track previous floor for collision detection
        self.state = "ROLLING"
        
        # Local coordinates (relative to track)
        self.local_x = -0.1 
        self.local_v = 0.0
        
        # Global coordinates
        self.global_x, self.global_y = 0.0, 0.0
        self.global_vx, self.global_vy = 0.0, 0.0

        # Initialize Height
        if self.current_floor_idx in self.lvl.floors:
            self.local_y = float(self.lvl.floors[self.current_floor_idx]['func'](self.local_x))
        else:
            self.current_floor_idx = -1
            self.local_y = 0.0
            self.state = "FALLING"

    def update(self, angle, d_angle_dt):
        dt_step = self.params.dt / self.params.sub_steps
        
        for _ in range(self.params.sub_steps):
            if self.state == "ROLLING":
                self._handle_rolling(dt_step, angle, d_angle_dt)
            elif self.state == "FALLING":
                self._handle_falling(dt_step, angle)
                
        return self.global_x, self.global_y

    def _handle_rolling(self, dt, angle, d_angle_dt):
        if self.current_floor_idx not in self.lvl.floors:
            self.state = "FALLING"
            return

        floor = self.lvl.floors[self.current_floor_idx]
        slope_val = float(floor['slope'](self.local_x))
        alpha = np.arctan(slope_val)
        
        # Physics Equation: Gravity component + Friction
        accel = -self.params.gravity * np.sin(alpha + angle) - self.params.friction * self.local_v
        
        self.local_v += accel * dt
        self.local_x += self.local_v * dt
        self.local_y = float(floor['func'](self.local_x))
        
        self.global_x, self.global_y = rotate_vector(self.local_x, self.local_y, angle)
        
        # Check if ball rolled off the edge
        if self.local_x < floor['min'] or self.local_x > floor['max']:
            self._transition_to_falling(slope_val, angle, d_angle_dt)

    def _transition_to_falling(self, slope, angle, d_angle_dt):
        self.previous_floor_idx = self.current_floor_idx  # Remember previous floor
        self.state = "FALLING"
        
        # Convert local velocity to global velocity
        vx_loc = self.local_v
        vy_loc = slope * self.local_v
        
        gvx_rel, gvy_rel = rotate_vector(vx_loc, vy_loc, angle)
        
        # Add tangential velocity from the platform rotation
        vx_tan = -d_angle_dt * self.global_y
        vy_tan = d_angle_dt * self.global_x
        
        self.global_vx = gvx_rel + vx_tan
        self.global_vy = gvy_rel + vy_tan

    def _handle_falling(self, dt, angle):
        # Gravity
        self.global_vy -= self.params.gravity * dt
        self.global_x += self.global_vx * dt
        self.global_y += self.global_vy * dt
        
        # Collision Detection - ONLY check floors ahead of previous floor
        lx_chk, ly_chk = inverse_rotate_vector(self.global_x, self.global_y, angle)
        
        best_collision = None
        max_target_y = float('-inf')  # Find floor with highest y (most below the ball)
        
        # Determine which floors to check
        if self.previous_floor_idx >= 0 and self.previous_floor_idx in self.lvl.floors:
            # ONLY check floors with index > previous_floor_idx (forward progression)
            floors_to_check = [i for i in sorted(self.lvl.floors.keys()) if i > self.previous_floor_idx]
        else:
            # If no previous floor, check all floors
            floors_to_check = sorted(self.lvl.floors.keys())
        
        for f_idx in floors_to_check:
            floor = self.lvl.floors[f_idx]
            
            # Check X bounds
            if floor['min'] <= lx_chk <= floor['max']:
                target_y = float(floor['func'](lx_chk))
                
                # Check Y collision - ball is below or at the floor
                if ly_chk <= target_y:
                    # Calculate if ball is moving toward this floor
                    slope = float(floor['slope'](lx_chk))
                    floor_angle = np.arctan(slope) + angle
                    nx, ny = -np.sin(floor_angle), np.cos(floor_angle)
                    v_dot_n = self.global_vx * nx + self.global_vy * ny
                    
                    # Only consider collision if ball is moving toward the floor
                    # And prefer floors that are more below the ball (higher target_y)
                    if v_dot_n < 0 and target_y > max_target_y:
                        max_target_y = target_y
                        best_collision = (f_idx, floor, lx_chk, ly_chk, target_y)
        
        # Resolve the best collision found
        if best_collision is not None:
            f_idx, floor, lx_chk, ly_chk, target_y = best_collision
            self._resolve_collision(lx_chk, ly_chk, target_y, floor, angle, f_idx)

    def _resolve_collision(self, lx, ly, target_y, floor, angle, f_idx):
        slope = float(floor['slope'](lx))
        floor_angle = np.arctan(slope) + angle
        
        nx, ny = -np.sin(floor_angle), np.cos(floor_angle)
        v_dot_n = self.global_vx * nx + self.global_vy * ny
        
        if v_dot_n < 0:
            # Bounce
            j = -(1 + self.params.restitution) * v_dot_n
            self.global_vx += j * nx
            self.global_vy += j * ny
            
            # Penetration fix
            fix = target_y - ly + 0.001
            self.global_x += nx * fix
            self.global_y += ny * fix

            # Stick to floor if impact is low
            if abs(v_dot_n) < 0.5:
                self.state = "ROLLING"
                self.previous_floor_idx = self.current_floor_idx
                self.current_floor_idx = f_idx
                self.local_x = lx
                self.local_y = target_y
                
                # Project global velocity back to local tangent
                tx, ty = np.cos(floor_angle), np.sin(floor_angle)
                self.local_v = self.global_vx * tx + self.global_vy * ty


class FuzzyController:
    """
    Fuzzy controller implementation based on MATLAB fuzzy logic system.
    Input 1: error (9 membership functions)
    Input 2: velocidad (5 membership functions)
    Output: inclinacion (9 membership functions)
    """
    def __init__(self):
        # Input 1: error - 9 membership functions
        self.error_labels = ['grandeNeg', 'medioNeg', 'pequeñoNeg', 'muyPequeñoNeg', 'cero',
                             'muyPequeñoPos', 'pequeñoPos', 'medioPos', 'grandePos']
        self.error_range = [-0.23, 0.23]
        self.error_mfs = {
            'grandeNeg': [-0.278, -0.23, -0.182],
            'medioNeg': [-0.22, -0.173, -0.125],
            'pequeñoNeg': [-0.163, -0.115, -0.06],
            'muyPequeñoNeg': [-0.08, -0.04, -0.005],
            'cero': [-0.005, 0, 0.005],
            'muyPequeñoPos': [0.005, 0.04, 0.08],
            'pequeñoPos': [0.06, 0.115, 0.163],
            'medioPos': [0.125, 0.173, 0.22],
            'grandePos': [0.182, 0.23, 0.278]
        }
        
        # Input 2: velocidad - 5 membership functions
        self.velocidad_labels = ['rapidaNeg', 'lentaNeg', 'cero', 'lentaPos', 'rapidaPos']
        self.velocidad_range = [-0.5, 0.5]
        self.velocidad_mfs = {
            'rapidaNeg': [-0.708333333333333, -0.5, -0.291666666666667],
            'lentaNeg': [-0.458333333333333, -0.25, -0.0416666666666667],
            'cero': [-0.208333333333333, 0, 0.208333333333333],
            'lentaPos': [0.0416666666666667, 0.25, 0.458333333333333],
            'rapidaPos': [0.291666666666667, 0.5, 0.708333333333333]
        }
        
        # Output: inclinacion - 9 membership functions
        self.inclinacion_labels = ['giraMuchoNeg', 'giraMedioNeg', 'giraPocoNeg', 'giraMuyPocoNeg', 'cero',
                                   'giraMuyPocoPos', 'giraPocoPos', 'giraMedioPos', 'giraMuchoPos']
        self.inclinacion_range = [-0.4, 0.4]
        self.inclinacion_mfs = {
            'giraMuchoNeg': [-0.483333333333333, -0.4, -0.316666666666667],
            'giraMedioNeg': [-0.383333333333333, -0.3, -0.216666666666667],
            'giraPocoNeg': [-0.283333333333333, -0.2, -0.116666666666667],
            'giraMuyPocoNeg': [-0.183333333333333, -0.1, -0.0166666666666667],
            'cero': [-0.0833333333333334, -2.77555756156289e-17, 0.0833333333333333],
            'giraMuyPocoPos': [0.0166666666666666, 0.1, 0.183333333333333],
            'giraPocoPos': [0.116666666666667, 0.2, 0.283333333333333],
            'giraMedioPos': [0.216666666666667, 0.3, 0.383333333333333],
            'giraMuchoPos': [0.316666666666667, 0.4, 0.483333333333333]
        }
        
        # Rules: format is (error_mf_idx, velocidad_mf_idx, inclinacion_mf_idx)
        # MATLAB uses 1-based indexing, we convert to 0-based
        # Rule format: "1 3, 1 (1) : 1" means error=MF1, velocidad=MF3, output=MF1
        self.rules = [
            (0, 2, 0),   # 1 3, 1 (1) : 1 -> error=grandeNeg, velocidad=cero, output=giraMuchoNeg
            (1, 2, 1),   # 2 3, 2 (1) : 1 -> error=medioNeg, velocidad=cero, output=giraMedioNeg
            (2, 2, 2),   # 3 3, 3 (1) : 1 -> error=pequeñoNeg, velocidad=cero, output=giraPocoNeg
            (3, 2, 3),   # 4 3, 4 (1) : 1 -> error=muyPequeñoNeg, velocidad=cero, output=giraMuyPocoNeg
            (5, 2, 5),   # 6 3, 6 (1) : 1 -> error=muyPequeñoPos, velocidad=cero, output=giraMuyPocoPos
            (6, 2, 6),   # 7 3, 7 (1) : 1 -> error=pequeñoPos, velocidad=cero, output=giraPocoPos
            (7, 2, 7),   # 8 3, 8 (1) : 1 -> error=medioPos, velocidad=cero, output=giraMedioPos
            (8, 2, 8),   # 9 3, 9 (1) : 1 -> error=grandePos, velocidad=cero, output=giraMuchoPos
            (0, 1, 0),   # 1 2, 1 (1) : 1 -> error=grandeNeg, velocidad=lentaNeg, output=giraMuchoNeg
            (1, 1, 1),   # 2 2, 2 (1) : 1 -> error=medioNeg, velocidad=lentaNeg, output=giraMedioNeg
            (2, 1, 2),   # 3 2, 3 (1) : 1 -> error=pequeñoNeg, velocidad=lentaNeg, output=giraPocoNeg
            (6, 3, 6),   # 7 4, 7 (1) : 1 -> error=pequeñoPos, velocidad=lentaPos, output=giraPocoPos
            (7, 3, 7),   # 8 4, 8 (1) : 1 -> error=medioPos, velocidad=lentaPos, output=giraMedioPos
            (8, 3, 8),   # 9 4, 9 (1) : 1 -> error=grandePos, velocidad=lentaPos, output=giraMuchoPos
            (3, 0, 3),   # 4 1, 4 (1) : 1 -> error=muyPequeñoNeg, velocidad=rapidaNeg, output=giraMuyPocoNeg
            (5, 4, 5),   # 6 5, 6 (1) : 1 -> error=muyPequeñoPos, velocidad=rapidaPos, output=giraMuyPocoPos
            (2, 0, 2),   # 3 1, 3 (1) : 1 -> error=pequeñoNeg, velocidad=rapidaNeg, output=giraPocoNeg
            (6, 4, 6),   # 7 5, 7 (1) : 1 -> error=pequeñoPos, velocidad=rapidaPos, output=giraPocoPos
            (4, 4, 3),   # 5 5, 4 (1) : 1 -> error=cero, velocidad=rapidaPos, output=giraMuyPocoNeg
            (4, 0, 5),   # 5 1, 6 (1) : 1 -> error=cero, velocidad=rapidaNeg, output=giraMuyPocoPos
        ]

    def _trimf(self, x, abc):
        """Triangular membership function."""
        a, b, c = abc
        if x <= a or x >= c:
            return 0.0
        if x < b:
            return (x - a) / (b - a + 1e-9)
        else:
            return (c - x) / (c - b + 1e-9)

    def _fuzzify_error(self, error_val):
        """Fuzzify error input."""
        memberships = {}
        for label in self.error_labels:
            memberships[label] = self._trimf(error_val, self.error_mfs[label])
        return memberships

    def _fuzzify_velocidad(self, velocidad_val):
        """Fuzzify velocidad input."""
        memberships = {}
        for label in self.velocidad_labels:
            memberships[label] = self._trimf(velocidad_val, self.velocidad_mfs[label])
        return memberships

    def _centroid_defuzzify(self, aggregated_output):
        """
        Centroid defuzzification method.
        aggregated_output: dict mapping output labels to their aggregated membership values
        """
        numerator = 0.0
        denominator = 0.0
        
        # Use fine resolution for centroid calculation
        output_range = np.linspace(self.inclinacion_range[0], self.inclinacion_range[1], 1000)
        
        for x in output_range:
            # Calculate membership value at x for each output MF
            mu_x = 0.0
            for label in self.inclinacion_labels:
                mu_mf = self._trimf(x, self.inclinacion_mfs[label])
                # Aggregate using max (as per AggMethod='max')
                mu_x = max(mu_x, min(mu_mf, aggregated_output.get(label, 0.0)))
            
            numerator += x * mu_x
            denominator += mu_x
        
        if denominator == 0:
            return 0.0
        return numerator / denominator

    def compute(self, error, velocidad):
        """
        Compute fuzzy controller output.
        Args:
            error: error value (position error)
            velocidad: velocity value
        Returns:
            output: defuzzified output (inclinacion)
            debug: debug information dictionary
        """
        # Clamp inputs to ranges
        error = np.clip(error, self.error_range[0], self.error_range[1])
        velocidad = np.clip(velocidad, self.velocidad_range[0], self.velocidad_range[1])
        
        # 1. Fuzzification
        m_error = self._fuzzify_error(error)
        m_velocidad = self._fuzzify_velocidad(velocidad)
        
        # 2. Rule evaluation and aggregation
        # Initialize aggregated output (using max aggregation)
        aggregated_output = {label: 0.0 for label in self.inclinacion_labels}
        active_rules = []
        
        for error_mf_idx, velocidad_mf_idx, inclinacion_mf_idx in self.rules:
            # Get membership values
            error_label = self.error_labels[error_mf_idx]
            velocidad_label = self.velocidad_labels[velocidad_mf_idx]
            inclinacion_label = self.inclinacion_labels[inclinacion_mf_idx]
            
            # AND method: min
            rule_strength = min(m_error[error_label], m_velocidad[velocidad_label])
            
            if rule_strength > 0:
                # Implication method: min (clip output MF by rule strength)
                # Aggregation method: max (take maximum of all rule outputs)
                aggregated_output[inclinacion_label] = max(
                    aggregated_output[inclinacion_label],
                    rule_strength  # min implication
                )
                
                # Debug info
                active_rules.append({
                    'error_mf': error_label,
                    'velocidad_mf': velocidad_label,
                    'inclinacion_mf': inclinacion_label,
                    'strength': rule_strength,
                    'indices': (velocidad_mf_idx, error_mf_idx)
                })
        
        # 3. Defuzzification: centroid
        output = self._centroid_defuzzify(aggregated_output)
        
        return output, {
            'm_error': m_error,
            'm_velocidad': m_velocidad,
            'rules': active_rules,
            'output': output,
            'aggregated_output': aggregated_output
        }
        
class Dashboard:
    def __init__(self, level: Level, fuzzy: FuzzyController, history: Dict, config: SimulationConfig):
        self.lvl = level
        self.fuzzy = fuzzy
        self.hist = history
        self.config = config
        # Larger figure with better proportions
        self.fig = plt.figure(figsize=(24, 16))
        # Better grid: 8 rows x 5 columns with improved spacing
        self.gs = self.fig.add_gridspec(6, 5, hspace=1, wspace=0.4, 
                                        left=0.06, right=0.97, top=0.95, bottom=0.05)
        self.artists = []
        
        self._setup_track_view()
        self._setup_heatmap()
        self._setup_defuzz_view()
        self._setup_state_info()
        self._setup_setpoint_plot()
        self._setup_velocity_plot()
        self._setup_control_output_plot()
        self._setup_platform_angle_plot()
        self._setup_phase_plot()
        self._setup_fuzz_pos()
        self._setup_fuzz_vel()
        self._setup_output_mf_view()

    def _setup_track_view(self):
        ax = self.fig.add_subplot(self.gs[0:3, 0:2])
        ax.set_title("Simulation Track", fontsize=14, fontweight='bold', pad=10)
        ax.set_xlim(-0.5, 0.5); ax.set_ylim(-0.5, 0.5); ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("X Position (m)", fontsize=11)
        ax.set_ylabel("Y Position (m)", fontsize=11)
        
        # Plot all setpoints for current level
        finales = np.zeros((4, 7, 2))
        finales[0, :, 0] = [0, 0, 0, 0, 0, 0, -0.02554]
        finales[0, :, 1] = [0.11429, 0.06857, 0.02286, -0.02286, -0.06857, -0.11429, -0.16035]
        finales[1, :, 0] = [0.06211, -0.04758, 0.04969, -0.04847, 0.04406, -0.05409, -0.02554]
        finales[1, :, 1] = [0.11223, 0.06799, 0.02154, -0.02411, -0.06961, -0.11585, -0.16035]
        finales[2, :, 0] = [0.12394, -0.09503, 0.09924, -0.0968, 0.08803, -0.108, -0.02554]
        finales[2, :, 1] = [0.10605, 0.06374, 0.01759, -0.02787, -0.07271, -0.12053, -0.16035]
        finales[3, :, 0] = [0.12394, 0.14224, 0.14851, -0.14488, 0.1318, -0.108, -0.02554]
        finales[3, :, 1] = [0.10605, 0.05771, 0.01102, -0.03412, -0.07789, -0.12053, -0.16035]
        
        nivel_idx = self.config.nivel - 1
        if nivel_idx < 0 or nivel_idx >= 4: nivel_idx = 0
        setpoints_x = finales[nivel_idx, :, 0]
        setpoints_y = finales[nivel_idx, :, 1]
        self.setpoint_markers, = ax.plot(setpoints_x, setpoints_y, 'g*', markersize=12, 
                                         label='Setpoints', zorder=5, alpha=0.7)
        self.current_setpoint_marker, = ax.plot([], [], 'gX', markersize=15, 
                                                label='Current Target', zorder=6, markeredgewidth=2)
        
        self.track_lines = [ax.plot([], [], 'k-', lw=1.5)[0] for _ in self.lvl.visual_segments]
        self.ball_dot, = ax.plot([], [], 'ro', markersize=10, zorder=10, label='Ball')
        self.ball_trace, = ax.plot([], [], 'r-', lw=1, alpha=0.6, label='Trajectory')
        #ax.legend(loc='upper right', fontsize=8)

    def _setup_heatmap(self):
        ax = self.fig.add_subplot(self.gs[0:1, 2:3])
        ax.set_title("Active Rules Matrix", fontsize=12, fontweight='bold', pad=8)
        ax.set_xlabel("Error", fontsize=10); ax.set_ylabel("Velocidad", fontsize=10)
        # Error has 9 MFs, Velocidad has 5 MFs
        ax.set_xticks(range(9)); ax.set_xticklabels([l[:4] for l in self.fuzzy.error_labels], rotation=45, fontsize=8)
        ax.set_yticks(range(5)); ax.set_yticklabels([l[:4] for l in self.fuzzy.velocidad_labels], fontsize=8)
        self.heatmap_img = ax.imshow(np.zeros((5, 9)), cmap='Reds', vmin=0, vmax=1, origin='lower', aspect='auto')
        plt.colorbar(self.heatmap_img, ax=ax, fraction=0.046, pad=0.04)

    def _setup_defuzz_view(self):
        ax = self.fig.add_subplot(self.gs[2:3, 2:3])
        ax.set_title("Defuzzification", fontsize=12, fontweight='bold', pad=8)
        ax.set_xlim(-0.5, 0.5); ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Error", fontsize=9)
        # Get centers of output MFs
        output_centers = [self.fuzzy.inclinacion_mfs[label][1] for label in self.fuzzy.inclinacion_labels]
        for i, (label, center) in enumerate(zip(self.fuzzy.inclinacion_labels, output_centers)):
            ax.axvline(center, color='gray', linestyle=':', alpha=0.5)
            ax.text(center, 1.02, label[:4], ha='center', fontsize=7, rotation=45)
        self.out_bars = ax.bar(output_centers, [0]*9, width=0.02, color='blue', alpha=0.6)
        self.defuzz_out_line = ax.axvline(0, color='red', lw=2)
        self.output_centers = output_centers

    def _setup_fuzz_plot(self, gs_pos, title, mf_dict, mf_labels, mf_range, color):
        ax = self.fig.add_subplot(gs_pos)
        ax.set_title(title, fontsize=11, fontweight='bold', pad=6)
        ax.set_ylim(0, 1.1)
        ax.set_xlim(mf_range[0] * 1.2, mf_range[1] * 1.2)
        ax.set_xlabel(title.split(':')[-1].strip(), fontsize=9)
        ax.set_ylabel("Membership", fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Draw static MF triangles
        x_static = np.linspace(mf_range[0] * 1.2, mf_range[1] * 1.2, 200)
        for label in mf_labels:
            abc = mf_dict[label]
            y = [self.fuzzy._trimf(xi, abc) for xi in x_static]
            ax.plot(x_static, y, 'k-', lw=0.5, alpha=0.5)
            ax.fill_between(x_static, 0, y, alpha=0.05, color=color)
        
        line = ax.axvline(0, color='red', lw=1.5)
        dots, = ax.plot([], [], f'{color[0]}o', markersize=4)
        return line, dots

    def _setup_setpoint_plot(self):
        ax = self.fig.add_subplot(self.gs[3:4, 0:3])
        ax.set_title("Position Tracking", fontsize=12, fontweight='bold', pad=8)
        ax.set_xlabel("Time Step", fontsize=10)
        ax.set_ylabel("Position (m)", fontsize=10)
        ax.grid(True, alpha=0.3)
        self.setpoint_line, = ax.plot([], [], 'g--', lw=2, label='Setpoint', alpha=0.8, marker='o', markersize=3, markevery=10)
        self.position_line, = ax.plot([], [], 'b-', lw=1.5, label='Position', alpha=0.8)
        self.error_line, = ax.plot([], [], 'r-', lw=1.2, label='Error', alpha=0.7)
        ax.legend(loc='best', fontsize=9)
        ax.set_xlim(0, len(self.hist['x']))
        self.setpoint_ax = ax

    def _setup_velocity_plot(self):
        ax = self.fig.add_subplot(self.gs[4:5, 0:1])
        ax.set_title("Velocity", fontsize=11, fontweight='bold', pad=6)
        ax.set_xlabel("Time Step", fontsize=9)
        ax.set_ylabel("Velocity (m/s)", fontsize=9)
        ax.grid(True, alpha=0.3)
        self.velocity_line, = ax.plot([], [], 'm-', lw=1.5, label='Global Vx', alpha=0.8)
        self.local_velocity_line, = ax.plot([], [], 'c--', lw=1.2, label='Local V', alpha=0.7)
        ax.legend(loc='best', fontsize=8)
        ax.set_xlim(0, len(self.hist['x']))
        self.velocity_ax = ax

    def _setup_control_output_plot(self):
        ax = self.fig.add_subplot(self.gs[4:5, 1:2])
        ax.set_title("Control Output", fontsize=11, fontweight='bold', pad=6)
        ax.set_xlabel("Time Step", fontsize=9)
        ax.set_ylabel("Inclinación (rad)", fontsize=9)
        ax.grid(True, alpha=0.3)
        self.control_output_line, = ax.plot([], [], 'orange', lw=1.5, label='Fuzzy Output', alpha=0.8)
        ax.axhline(0, color='k', linestyle=':', alpha=0.3)
        ax.legend(loc='best', fontsize=8)
        ax.set_xlim(0, len(self.hist['x']))
        self.control_output_ax = ax

    def _setup_platform_angle_plot(self):
        ax = self.fig.add_subplot(self.gs[4:5, 2:3])
        ax.set_title("Platform Angle", fontsize=11, fontweight='bold', pad=6)
        ax.set_xlabel("Time Step", fontsize=9)
        ax.set_ylabel("Angle (rad)", fontsize=9)
        ax.grid(True, alpha=0.3)
        self.angle_line, = ax.plot([], [], 'purple', lw=1.5, label='Platform Angle', alpha=0.8)
        self.angle_deg_line, = ax.plot([], [], 'brown', lw=1.2, linestyle='--', label='Angle (deg)', alpha=0.7)
        ax.axhline(0, color='k', linestyle=':', alpha=0.3)
        ax.legend(loc='best', fontsize=8)
        ax.set_xlim(0, len(self.hist['x']))
        self.angle_ax = ax

    def _setup_phase_plot(self):
        ax = self.fig.add_subplot(self.gs[4:5, 3:4])
        ax.set_title("Phase Plot", fontsize=11, fontweight='bold', pad=6)
        ax.set_xlabel("Error (m)", fontsize=9)
        ax.set_ylabel("Velocity (m/s)", fontsize=9)
        ax.grid(True, alpha=0.3)
        self.phase_line, = ax.plot([], [], 'b-', lw=1, alpha=0.6, label='Trajectory')
        self.phase_dot, = ax.plot([], [], 'ro', markersize=6, label='Current', zorder=10)
        ax.axhline(0, color='k', linestyle=':', alpha=0.3, linewidth=0.5)
        ax.axvline(0, color='k', linestyle=':', alpha=0.3, linewidth=0.5)
        ax.legend(loc='best', fontsize=8)
        self.phase_ax = ax

    def _setup_state_info(self):
        ax = self.fig.add_subplot(self.gs[1:2, 3:5])
        ax.axis('off')
        ax.set_title("System State", fontsize=12, fontweight='bold', pad=-30, loc='left')
        self.state_text = ax.text(0.1, 0.9, '', transform=ax.transAxes, fontsize=10,
                                  verticalalignment='top', family='monospace',
                                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6, pad=8))

    def _setup_fuzz_pos(self):
        self.line_p, self.dots_p = self._setup_fuzz_plot(
            self.gs[5:6, 0:1], "Fuzzification: Error", 
            self.fuzzy.error_mfs, self.fuzzy.error_labels, 
            self.fuzzy.error_range, 'blue'
        )

    def _setup_fuzz_vel(self):
        self.line_v, self.dots_v = self._setup_fuzz_plot(
            self.gs[5:6, 1:2], "Fuzzification: Velocidad", 
            self.fuzzy.velocidad_mfs, self.fuzzy.velocidad_labels,
            self.fuzzy.velocidad_range, 'green'
        )

    def _setup_output_mf_view(self):
        ax = self.fig.add_subplot(self.gs[5:6, 2:3])
        ax.set_title("Output MFs: Inclinación", fontsize=11, fontweight='bold', pad=6)
        ax.set_ylim(0, 1.1)
        ax.set_xlim(self.fuzzy.inclinacion_range[0] * 1.2, self.fuzzy.inclinacion_range[1] * 1.2)
        ax.set_xlabel("Inclinación (rad)", fontsize=9)
        ax.set_ylabel("Membership", fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Draw static output MF triangles
        x_static = np.linspace(self.fuzzy.inclinacion_range[0] * 1.2, 
                              self.fuzzy.inclinacion_range[1] * 1.2, 200)
        for label in self.fuzzy.inclinacion_labels:
            abc = self.fuzzy.inclinacion_mfs[label]
            y = [self.fuzzy._trimf(xi, abc) for xi in x_static]
            ax.plot(x_static, y, 'k-', lw=0.5, alpha=0.5)
            ax.fill_between(x_static, 0, y, alpha=0.05, color='orange')
        
        self.output_line = ax.axvline(0, color='red', lw=2, label='Output')
        ax.legend(loc='upper right', fontsize=8)

    def update_frame(self, f):
        # 1. Update Track Geometry
        ang = self.hist['angle'][f]
        for ln, (lx, ly) in zip(self.track_lines, self.lvl.visual_segments):
            gx, gy = rotate_vector(lx, ly, ang)
            ln.set_data(gx, gy)
        
        self.ball_dot.set_data([self.hist['x'][f]], [self.hist['y'][f]])
        trace_len = min(200, f+1)
        self.ball_trace.set_data(self.hist['x'][max(0, f-trace_len+1):f+1], 
                                 self.hist['y'][max(0, f-trace_len+1):f+1])
        
        # Update current setpoint marker
        if f < len(self.hist['setpoint']):
            current_setpoint_x = self.hist['setpoint'][f]
            # Find corresponding Y from finales
            finales = np.zeros((4, 7, 2))
            finales[0, :, 0] = [0, 0, 0, 0, 0, 0, -0.02554]
            finales[0, :, 1] = [0.11429, 0.06857, 0.02286, -0.02286, -0.06857, -0.11429, -0.16035]
            finales[1, :, 0] = [0.06211, -0.04758, 0.04969, -0.04847, 0.04406, -0.05409, -0.02554]
            finales[1, :, 1] = [0.11223, 0.06799, 0.02154, -0.02411, -0.06961, -0.11585, -0.16035]
            finales[2, :, 0] = [0.12394, -0.09503, 0.09924, -0.0968, 0.08803, -0.108, -0.02554]
            finales[2, :, 1] = [0.10605, 0.06374, 0.01759, -0.02787, -0.07271, -0.12053, -0.16035]
            finales[3, :, 0] = [0.12394, 0.14224, 0.14851, -0.14488, 0.1318, -0.108, -0.02554]
            finales[3, :, 1] = [0.10605, 0.05771, 0.01102, -0.03412, -0.07789, -0.12053, -0.16035]
            nivel_idx = self.config.nivel - 1
            if nivel_idx < 0 or nivel_idx >= 4: nivel_idx = 0
            # Find closest setpoint
            setpoints_x = finales[nivel_idx, :, 0]
            setpoints_y = finales[nivel_idx, :, 1]
            closest_idx = np.argmin(np.abs(setpoints_x - current_setpoint_x))
            self.current_setpoint_marker.set_data([setpoints_x[closest_idx]], [setpoints_y[closest_idx]])

        # 2. Update Setpoint Plot
        time_steps = list(range(f+1))
        self.setpoint_line.set_data(time_steps, self.hist['setpoint'][:f+1])
        self.position_line.set_data(time_steps, self.hist['x'][:f+1])
        self.error_line.set_data(time_steps, self.hist['error'][:f+1])
        
        # Update y-axis range dynamically
        if f >= 0 and len(self.hist['x']) > 0:
            all_vals = self.hist['x'][:f+1] + self.hist['setpoint'][:f+1] + self.hist['error'][:f+1]
            if all_vals:
                y_min, y_max = min(all_vals), max(all_vals)
                y_range = y_max - y_min
                padding = max(0.1 * y_range, 0.05) if y_range > 0 else 0.1
                self.setpoint_ax.set_ylim(y_min - padding, y_max + padding)

        # 3. Update Velocity Plot
        self.velocity_line.set_data(time_steps, self.hist['global_vx'][:f+1])
        self.local_velocity_line.set_data(time_steps, self.hist['lv'][:f+1])
        if f >= 0 and len(self.hist['global_vx']) > 0:
            all_vel = self.hist['global_vx'][:f+1] + self.hist['lv'][:f+1]
            if all_vel:
                v_min, v_max = min(all_vel), max(all_vel)
                v_range = v_max - v_min
                padding = max(0.1 * v_range, 0.01) if v_range > 0 else 0.1
                self.velocity_ax.set_ylim(v_min - padding, v_max + padding)

        # 4. Update Control Output Plot
        control_outputs = [d['output'] if d else 0.0 for d in self.hist['debug'][:f+1]]
        self.control_output_line.set_data(time_steps, control_outputs)
        if control_outputs:
            c_min, c_max = min(control_outputs), max(control_outputs)
            c_range = c_max - c_min
            padding = max(0.1 * c_range, 0.01) if c_range > 0 else 0.1
            self.control_output_ax.set_ylim(c_min - padding, c_max + padding)

        # 5. Update Platform Angle Plot
        angles = self.hist['angle'][:f+1]
        angles_deg = [np.degrees(a) for a in angles]
        self.angle_line.set_data(time_steps, angles)
        self.angle_deg_line.set_data(time_steps, angles_deg)
        if angles:
            a_min, a_max = min(angles), max(angles)
            a_range = a_max - a_min
            padding = max(0.1 * a_range, 0.01) if a_range > 0 else 0.1
            self.angle_ax.set_ylim(a_min - padding, a_max + padding)

        # 6. Update Phase Plot
        phase_trace_len = min(500, f+1)
        phase_errors = self.hist['error'][max(0, f-phase_trace_len+1):f+1]
        phase_vels = self.hist['global_vx'][max(0, f-phase_trace_len+1):f+1]
        self.phase_line.set_data(phase_errors, phase_vels)
        if f < len(self.hist['error']):
            self.phase_dot.set_data([self.hist['error'][f]], [self.hist['global_vx'][f]])
        if phase_errors and phase_vels:
            self.phase_ax.set_xlim(min(phase_errors) - 0.01, max(phase_errors) + 0.01)
            self.phase_ax.set_ylim(min(phase_vels) - 0.01, max(phase_vels) + 0.01)

        # 7. Update State Info
        current_floor = self.hist.get('current_floor', [0] * len(self.hist['x']))[f] if f < len(self.hist.get('current_floor', [])) else 0
        state_text = f"Level: {self.config.nivel}\n"
        state_text += f"Floor: {current_floor}\n"
        state_text += f"Position: {self.hist['x'][f]:.4f} m\n"
        state_text += f"Error: {self.hist['error'][f]:.4f} m\n"
        state_text += f"Velocity: {self.hist['global_vx'][f]:.4f} m/s\n"
        state_text += f"Angle: {np.degrees(self.hist['angle'][f]):.2f}°\n"
        if f < len(self.hist['debug']) and self.hist['debug'][f]:
            state_text += f"Control Out: {self.hist['debug'][f]['output']:.4f} rad"
        self.state_text.set_text(state_text)

        # 8. Update Debug Visuals
        info = self.hist['debug'][f]
        if not info: return self._get_artists() # Skip if no debug data (falling)

        error_val = self.hist['error'][f]
        velocidad_val = self.hist['global_vx'][f]
        
        # Fuzzification Lines/Dots
        self.line_p.set_xdata([error_val])
        self.line_v.set_xdata([velocidad_val])
        error_mems = [info['m_error'][k] for k in self.fuzzy.error_labels]
        velocidad_mems = [info['m_velocidad'][k] for k in self.fuzzy.velocidad_labels]
        self.dots_p.set_data([error_val]*len(self.fuzzy.error_labels), error_mems)
        self.dots_v.set_data([velocidad_val]*len(self.fuzzy.velocidad_labels), velocidad_mems)

        # Heatmap: velocidad (rows) x error (columns)
        grid = np.zeros((5, 9))
        for r in info['rules']:
            velocidad_idx = self.fuzzy.velocidad_labels.index(r['velocidad_mf'])
            error_idx = self.fuzzy.error_labels.index(r['error_mf'])
            grid[velocidad_idx, error_idx] = max(grid[velocidad_idx, error_idx], r['strength'])
        self.heatmap_img.set_data(grid)

        # Defuzzification Bars
        bar_h = [0] * 9
        for r in info['rules']:
            inclinacion_idx = self.fuzzy.inclinacion_labels.index(r['inclinacion_mf'])
            bar_h[inclinacion_idx] = max(bar_h[inclinacion_idx], r['strength'])
        
        for bar, h in zip(self.out_bars, bar_h): bar.set_height(h)
        self.defuzz_out_line.set_xdata([info['output']])
        self.output_line.set_xdata([info['output']])
        
        return self._get_artists()

    def _get_artists(self):
        return (self.track_lines + [self.ball_dot, self.ball_trace, self.setpoint_markers, 
                                   self.current_setpoint_marker, self.setpoint_line, 
                                   self.position_line, self.error_line, self.velocity_line,
                                   self.local_velocity_line, self.control_output_line,
                                   self.angle_line, self.angle_deg_line, self.phase_line,
                                   self.phase_dot, self.line_p, self.dots_p, 
                                   self.line_v, self.dots_v, self.heatmap_img, self.defuzz_out_line,
                                   self.output_line, self.state_text] + list(self.out_bars))

    def show(self):
        ani = FuncAnimation(self.fig, self.update_frame, frames=len(self.hist['x']), interval=20, blit=False)
        plt.show()

def controller(piso, posX, nivel):
    """
    Calculate error based on current position, floor, and level.
    Args:
        piso: current floor index (1-7, MATLAB 1-based, we use 0-based internally)
        posX: current X position
        nivel: current level (1-4, MATLAB 1-based, we use 0-based internally)
    Returns:
        error: position error (posX - target_x)
    """
    # Initialize finales matrix: [Nivel x Piso x Coordenada(X,Y)]
    # Note: MATLAB uses 1-based indexing, Python uses 0-based
    finales = np.zeros((4, 7, 2))
    
    # --- CONFIGURACIÓN DE METAS (X, Y) ---
    
    # NIVEL 1 (index 0)
    finales[0, :, 0] = [0, 0, 0, 0, 0, 0, -0.02554]  # Valores X
    finales[0, :, 1] = [0.11429, 0.06857, 0.02286, -0.02286, -0.06857, -0.11429, -0.16035]  # Valores Y
    
    # NIVEL 2 (index 1)
    finales[1, :, 0] = [0.06211, -0.04758, 0.04969, -0.04847, 0.04406, -0.05409, -0.02554]  # Fixed typo: 0.05a409 -> 0.05409
    finales[1, :, 1] = [0.11223, 0.06799, 0.02154, -0.02411, -0.06961, -0.11585, -0.16035]
    
    # NIVEL 3 (index 2)
    finales[2, :, 0] = [0.12394, -0.09503, 0.09924, -0.0968, 0.08803, -0.108, -0.02554]
    finales[2, :, 1] = [0.10605, 0.06374, 0.01759, -0.02787, -0.07271, -0.12053, -0.16035]
    
    # NIVEL 4 (index 3)
    finales[3, :, 0] = [0.12394, 0.14224, 0.14851, -0.14488, 0.1318, -0.108, -0.02554]
    finales[3, :, 1] = [0.10605, 0.05771, 0.01102, -0.03412, -0.07789, -0.12053, -0.16035]
    
    # --- EXTRACCIÓN Y CÁLCULO DEL ERROR ---
    
    # Convert from MATLAB 1-based to Python 0-based indexing
    nivel_idx = nivel - 1
    piso_idx = piso - 2
    
    # Bounds checking
    if nivel_idx < 0 or nivel_idx >= 4:
        nivel_idx = 0
    if piso_idx < 0 or piso_idx >= 7:
        piso_idx = 0
    
    # Get target X position
    end_x = finales[nivel_idx, piso_idx, 0]
    
    # Calculate error: posX - end_x
    error = posX - end_x
    
    return error

def main():
    # 1. Setup
    config = SimulationConfig()
    level = Level(config)
    ball = PhysicsBall(level, start_idx=1)
    fuzzy = FuzzyController()
    
    # 2. Simulation Loop
    history = {'x': [], 'y': [], 'angle': [], 'lx': [], 'lv': [], 'global_vx': [], 
               'debug': [], 'setpoint': [], 'error': [], 'current_floor': []}
    current_angle = 0.0
    
    # Simulation parameters for CSV export
    Ts = 0.033  # Sampling time in seconds
    simulation_duration = 30  # 10 seconds
    num_steps = int(simulation_duration / Ts)  # Number of steps for 10 seconds
    
    # Update physics dt to match simulation sampling time
    ball.params.dt = Ts
    
    print(f"Simulating for {simulation_duration} seconds with Ts = {Ts} s ({num_steps} steps)...")
    for step in range(num_steps):
        # Get current floor index (0-based) and convert to MATLAB 1-based piso
        piso = ball.current_floor_idx + 1 if ball.current_floor_idx >= 0 else 1
        
        # Calculate error using controller function
        error = controller(piso, ball.global_x, config.nivel)
        
        # Calculate target position (setpoint) for visualization
        # Reconstruct finales matrix to get target
        finales = np.zeros((4, 7, 2))
        finales[0, :, 0] = [0, 0, 0, 0, 0, 0, -0.02554]
        finales[1, :, 0] = [0.06211, -0.04758, 0.04969, -0.04847, 0.04406, -0.05409, -0.02554]
        finales[2, :, 0] = [0.12394, -0.09503, 0.09924, -0.0968, 0.08803, -0.108, -0.02554]
        finales[3, :, 0] = [0.12394, 0.14224, 0.14851, -0.14488, 0.1318, -0.108, -0.02554]
        nivel_idx = config.nivel - 1
        piso_idx = piso - 2
        if nivel_idx < 0 or nivel_idx >= 4: nivel_idx = 0
        if piso_idx < 0 or piso_idx >= 7: piso_idx = 0
        target_x = finales[nivel_idx, piso_idx, 0]
        
        # Compute fuzzy controller output (error, velocidad)
        # Note: velocidad is the velocity in X direction (global_vx)
        output, debug = fuzzy.compute(error, ball.global_vx)
        
        # Convert fuzzy output to target angle (output is already in correct range)
        # The output is inclinacion, which directly maps to tilt angle
        target_angle = np.clip(output, -np.radians(config.max_tilt), np.radians(config.max_tilt))
        

        # Kinematics Step
        angle_diff = target_angle - current_angle
        omega = np.clip(angle_diff / Ts, -config.max_omega, config.max_omega)
        new_angle = current_angle + omega * Ts
        
        # Physics Step
        gx, gy = ball.update(new_angle, omega)
        current_angle = new_angle
        
        # Record Data
        history['x'].append(gx)
        history['y'].append(gy)
        history['angle'].append(current_angle)
        history['lx'].append(ball.local_x)
        history['lv'].append(ball.local_v)
        history['global_vx'].append(ball.global_vx)
        history['setpoint'].append(target_x)
        history['error'].append(error)
        history['debug'].append(debug)
        history['current_floor'].append(piso)

    # 3. Save x and y data to CSV
    csv_filename = 'simulation_data.csv'
    print(f"Saving simulation data to {csv_filename}...")
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['time', 'x', 'y'])  # Header
        for i in range(len(history['x'])):
            time = i * Ts
            writer.writerow([time, history['x'][i], history['y'][i]])
    print(f"Data saved to {csv_filename}")

    # 4. Visualization
    dashboard = Dashboard(level, fuzzy, history, config)
    dashboard.show()

if __name__ == "__main__":
    main()