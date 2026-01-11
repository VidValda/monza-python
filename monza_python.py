import numpy as np
import json
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
    friction: float = 0.01
    restitution: float = 0.1  # Bounciness
    sub_steps: int = 10       # Physics accuracy

@dataclass
class SimulationConfig:
    diff_path: str = 'dificultad1.json'
    circ_path: str = 'circulos.json'
    duration_steps: int = 2000
    max_tilt: float = 45.0    # Degrees
    max_omega: float = 8.0    # Rad/s
    setpoint: float = 0.0      # Target position for fuzzy controller

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
        self.state = "ROLLING"
        
        # Local coordinates (relative to track)
        self.local_x = 0.01 
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
        
        # Collision Detection
        for f_idx, floor in self.lvl.floors.items():    
            lx_chk, ly_chk = inverse_rotate_vector(self.global_x, self.global_y, angle)
            
            # Check X bounds
            if floor['min'] <= lx_chk <= floor['max']:
                target_y = float(floor['func'](lx_chk))
                
                # Check Y collision (tunneling check simplified)
                if ly_chk <= target_y:
                    self._resolve_collision(lx_chk, ly_chk, target_y, floor, angle, f_idx)
                    if self.state == "ROLLING": break

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
                self.current_floor_idx = f_idx
                self.local_x = lx
                self.local_y = target_y
                
                # Project global velocity back to local tangent
                tx, ty = np.cos(floor_angle), np.sin(floor_angle)
                self.local_v = self.global_vx * tx + self.global_vy * ty


class FuzzyController:
    def __init__(self):
        self.labels = ['NB', 'NS', 'Z', 'PS', 'PB']
        
        # Membership Centers - expanded ranges for error control
        # Error = setpoint - position: positive error means position is left of setpoint
        self.sets_pos = {'NB': -0.5, 'NS': -0.25, 'Z': 0.0, 'PS': 0.25, 'PB': 0.5}
        self.sets_vel = {'NB': -0.5, 'NS': -0.25, 'Z': 0.0, 'PS': 0.25, 'PB': 0.5}
        # Output range matches max tilt (45 degrees = 0.785 radians)
        self.sets_out = {'NB': -0.785, 'NS': -0.4, 'Z': 0.0, 'PS': 0.4, 'PB': 0.785}
        
        # Rule Base: (Error, Velocity, Output)
        # For error control: positive error (position left of setpoint) needs positive output (tilt right)
        # Negative error (position right of setpoint) needs negative output (tilt left)
        # This is INVERTED from position control (error = setpoint - position)
        self.rules_def = [
            ('NB', 'NB', 'NB'), ('NB', 'NS', 'NB'), ('NB', 'Z',  'NB'), ('NB', 'PS', 'NB'), ('NB', 'PB', 'NS'), 
            ('NS', 'NB', 'NB'), ('NS', 'NS', 'NB'), ('NS', 'Z',  'NS'), ('NS', 'PS', 'NS'), ('NS', 'PB', 'Z'),
            ('Z',  'NB', 'NS'), ('Z',  'NS', 'NS'), ('Z',  'Z',  'Z'), ('Z',  'PS', 'PS'), ('Z',  'PB', 'PS'),
            ('PS', 'NB', 'Z'), ('PS', 'NS', 'PS'), ('PS', 'Z',  'PS'), ('PS', 'PS', 'PB'), ('PS', 'PB', 'PB'),
            ('PB', 'NB', 'PS'), ('PB', 'NS', 'PB'), ('PB', 'Z',  'PB'), ('PB', 'PS', 'PB'), ('PB', 'PB', 'PB'),
        ]

    def _trimf(self, x, abc):
        """Triangular membership function generator."""
        a, b, c = abc
        return max(min((x - a) / (b - a + 1e-9), (c - x) / (c - b + 1e-9)), 0)

    def _get_memberships(self, val, sets):
        """Calculates membership degree for all sets."""
        mems = {}
        vals = [sets[k] for k in self.labels]
        
        for i, k in enumerate(self.labels):
            center = vals[i]
            left = vals[i-1] if i > 0 else center - (vals[i+1]-center)
            right = vals[i+1] if i < len(vals)-1 else center + (center-vals[i-1])
            mems[k] = self._trimf(val, [left, center, right])
        return mems

    def compute(self, pos, vel):
        # 1. Fuzzification
        m_pos = self._get_memberships(pos, self.sets_pos)
        m_vel = self._get_memberships(vel, self.sets_vel)
        
        numerator, denominator = 0.0, 0.0
        active_rules = [] 
        
        # 2. Rule Evaluation
        for r_pos, r_vel, r_out in self.rules_def:
            strength = min(m_pos[r_pos], m_vel[r_vel])
            
            if strength > 0:
                center = self.sets_out[r_out]
                numerator += strength * center
                denominator += strength
                
                # Debug info
                p_idx = self.labels.index(r_pos)
                v_idx = self.labels.index(r_vel)
                active_rules.append({'indices': (v_idx, p_idx), 'str': strength, 'out': center})
                
        # 3. Defuzzification (Weighted Average)
        output = numerator / denominator if denominator != 0 else 0.0
            
        return output, {
            'm_pos': m_pos, 'm_vel': m_vel, 'rules': active_rules, 'output': output
        }
        
class Dashboard:
    def __init__(self, level: Level, fuzzy: FuzzyController, history: Dict):
        self.lvl = level
        self.fuzzy = fuzzy
        self.hist = history
        self.fig = plt.figure(figsize=(16, 10))
        self.gs = self.fig.add_gridspec(4, 3)
        self.artists = []
        
        self._setup_track_view()
        self._setup_heatmap()
        self._setup_defuzz_view()
        self._setup_setpoint_plot()
        self._setup_fuzz_pos()
        self._setup_fuzz_vel()
        plt.tight_layout()

    def _setup_track_view(self):
        ax = self.fig.add_subplot(self.gs[0:2, 0:2])
        ax.set_title("Simulation")
        ax.set_xlim(-0.5, 0.5); ax.set_ylim(-0.5, 0.5); ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        self.track_lines = [ax.plot([], [], 'k-', lw=1.5)[0] for _ in self.lvl.visual_segments]
        self.ball_dot, = ax.plot([], [], 'ro', markersize=8, zorder=10)
        self.ball_trace, = ax.plot([], [], 'r-', lw=0.5, alpha=0.5)

    def _setup_heatmap(self):
        ax = self.fig.add_subplot(self.gs[0, 2])
        ax.set_title("Active Rules Matrix")
        ax.set_xlabel("Pos Error"); ax.set_ylabel("Velocity")
        ax.set_xticks(range(5)); ax.set_xticklabels(self.fuzzy.labels)
        ax.set_yticks(range(5)); ax.set_yticklabels(self.fuzzy.labels)
        self.heatmap_img = ax.imshow(np.zeros((5, 5)), cmap='Reds', vmin=0, vmax=1, origin='lower')

    def _setup_defuzz_view(self):
        ax = self.fig.add_subplot(self.gs[1, 2])
        ax.set_title("Defuzzification")
        ax.set_xlim(-0.6, 0.6); ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3)
        for k, v in self.fuzzy.sets_out.items():
            ax.axvline(v, color='gray', linestyle=':', alpha=0.5)
            ax.text(v, 1.02, k, ha='center', fontsize=8)
        self.out_bars = ax.bar(list(self.fuzzy.sets_out.values()), [0]*5, width=0.05, color='blue', alpha=0.6)
        self.out_line = ax.axvline(0, color='red', lw=2)

    def _setup_fuzz_plot(self, gs_pos, title, set_dict, color):
        ax = self.fig.add_subplot(gs_pos)
        ax.set_title(title)
        ax.set_ylim(0, 1.1)
        
        # Draw static MF triangles
        x_static = np.linspace(-1, 1, 100)
        for k in self.fuzzy.labels:
            vals = [set_dict[x] for x in self.fuzzy.labels]
            i = self.fuzzy.labels.index(k)
            c = vals[i]
            l = vals[i-1] if i > 0 else c - (vals[i+1]-c)
            r = vals[i+1] if i < len(vals)-1 else c + (c-vals[i-1])
            y = [self.fuzzy._trimf(xi, [l, c, r]) for xi in x_static]
            ax.plot(x_static, y, 'k-', lw=0.5, alpha=0.5)
            ax.fill_between(x_static, 0, y, alpha=0.05, color=color)
        
        line = ax.axvline(0, color='red', lw=1.5)
        dots, = ax.plot([], [], f'{color[0]}o')
        return line, dots

    def _setup_setpoint_plot(self):
        ax = self.fig.add_subplot(self.gs[2, 0:2])
        ax.set_title("Position vs Setpoint")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Position")
        ax.grid(True, alpha=0.3)
        self.setpoint_line, = ax.plot([], [], 'g--', lw=2, label='Setpoint', alpha=0.7)
        self.position_line, = ax.plot([], [], 'b-', lw=1.5, label='Position')
        self.error_line, = ax.plot([], [], 'r-', lw=1, label='Error', alpha=0.6)
        ax.legend(loc='upper right')
        ax.set_xlim(0, len(self.hist['x']))
        if len(self.hist['x']) > 0:
            all_vals = self.hist['x'] + self.hist['setpoint'] + self.hist['error']
            if all_vals:
                y_min, y_max = min(all_vals), max(all_vals)
                y_range = y_max - y_min
                ax.set_ylim(y_min - 0.1*y_range, y_max + 0.1*y_range)

    def _setup_fuzz_pos(self):
        self.line_p, self.dots_p = self._setup_fuzz_plot(self.gs[3, 0], "Fuzz: Error (setpoint - pos)", self.fuzzy.sets_pos, 'blue')

    def _setup_fuzz_vel(self):
        self.line_v, self.dots_v = self._setup_fuzz_plot(self.gs[3, 1], "Fuzz: Vel (lv)", self.fuzzy.sets_vel, 'green')

    def update_frame(self, f):
        # 1. Update Track Geometry
        ang = self.hist['angle'][f]
        for ln, (lx, ly) in zip(self.track_lines, self.lvl.visual_segments):
            gx, gy = rotate_vector(lx, ly, ang)
            ln.set_data(gx, gy)
        
        self.ball_dot.set_data([self.hist['x'][f]], [self.hist['y'][f]])
        self.ball_trace.set_data(self.hist['x'][max(0, f-50):f], self.hist['y'][max(0, f-50):f])

        # 2. Update Setpoint Plot
        time_steps = list(range(f+1))
        self.setpoint_line.set_data(time_steps, self.hist['setpoint'][:f+1])
        self.position_line.set_data(time_steps, self.hist['x'][:f+1])
        self.error_line.set_data(time_steps, self.hist['error'][:f+1])

        # 3. Update Debug Visuals
        info = self.hist['debug'][f]
        if not info: return self._get_artists() # Skip if no debug data (falling)

        error_val = self.hist['error'][f]
        lv_val = self.hist['lv'][f]
        
        # Fuzzification Lines/Dots (using error instead of position)
        self.line_p.set_xdata([error_val])
        self.line_v.set_xdata([lv_val])
        self.dots_p.set_data([error_val]*5, [info['m_pos'][k] for k in self.fuzzy.labels])
        self.dots_v.set_data([lv_val]*5, [info['m_vel'][k] for k in self.fuzzy.labels])

        # Heatmap
        grid = np.zeros((5, 5))
        for r in info['rules']: grid[r['indices'][0], r['indices'][1]] = r['str']
        self.heatmap_img.set_data(grid)

        # Defuzzification Bars
        bar_h = [0] * 5
        vals = list(self.fuzzy.sets_out.values())
        for r in info['rules']:
             try:
                idx = vals.index(r['out'])
                bar_h[idx] = max(bar_h[idx], r['str'])
             except ValueError: pass
        
        for bar, h in zip(self.out_bars, bar_h): bar.set_height(h)
        self.out_line.set_xdata([info['output']])
        
        return self._get_artists()

    def _get_artists(self):
        return self.track_lines + [self.ball_dot, self.ball_trace, self.setpoint_line, 
                                   self.position_line, self.error_line, self.line_p, self.dots_p, 
                                   self.line_v, self.dots_v, self.heatmap_img, self.out_line] + list(self.out_bars)

    def show(self):
        ani = FuncAnimation(self.fig, self.update_frame, frames=len(self.hist['x']), interval=20, blit=False)
        plt.show()

def main():
    # 1. Setup
    config = SimulationConfig()
    level = Level(config)
    ball = PhysicsBall(level, start_idx=0)
    fuzzy = FuzzyController()
    
    # 2. Simulation Loop
    history = {'x': [], 'y': [], 'angle': [], 'lx': [], 'lv': [], 'debug': [], 'setpoint': [], 'error': []}
    current_angle = 0.0
    
    print("Simulating...")
    for _ in range(config.duration_steps):
        # Compute error (setpoint - position) for fuzzy controller
        error = config.setpoint - ball.global_x
        output, debug = fuzzy.compute(error, ball.global_vx)
        # Convert fuzzy output to target angle (output is already in correct range)
        target_angle = np.clip(output, -np.radians(config.max_tilt), np.radians(config.max_tilt))

        # Kinematics Step
        angle_diff = target_angle - current_angle
        omega = np.clip(angle_diff / 0.01, -config.max_omega, config.max_omega)
        new_angle = current_angle + omega * 0.01
        
        # Physics Step
        gx, gy = ball.update(new_angle, omega)
        current_angle = new_angle
        
        # Record Data
        history['x'].append(gx)
        history['y'].append(gy)
        history['angle'].append(current_angle)
        history['lx'].append(ball.local_x)
        history['lv'].append(ball.local_v)
        history['setpoint'].append(config.setpoint)
        history['error'].append(error)
        history['debug'].append(debug)

    # 3. Visualization
    dashboard = Dashboard(level, fuzzy, history)
    dashboard.show()

if __name__ == "__main__":
    main()