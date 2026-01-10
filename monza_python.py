import numpy as np
import json
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.interpolate import interp1d

class Level:
    def __init__(self, diff_path, circ_path):
        self.floors = {}
        self.visuals = []
        self._load_data(diff_path, circ_path)

    def _load_data(self, d_path, c_path):
        try:
            with open(d_path, 'r') as f: d_data = json.load(f)
            with open(c_path, 'r') as f: c_data = json.load(f)
        except FileNotFoundError:
            print(f"Error: Files {d_path} or {c_path} not found")
            return

        for i in range(0, 20):
            xk, yk = f'xp{i}', f'yp{i}'
            if i == 0: xk, yk = 'xp','yp'
            if xk in d_data and yk in d_data:
                x = np.array(d_data[xk]).flatten()
                y = np.array(d_data[yk]).flatten()
                idx = np.argsort(x)
                x, y = x[idx], y[idx]
                dx = np.gradient(x)
                dx[dx == 0] = 1e-9
                slope = np.gradient(y) / dx
                self.floors[i] = {
                    'func': interp1d(x, y, kind='cubic', fill_value="extrapolate"),
                    'slope': interp1d(x, slope, kind='linear', fill_value="extrapolate"),
                    'min': x[0],
                    'max': x[-1]
                }
                self.visuals.append((x, y))

        i = 1
        while True:
            xk, yk = f'xl{i}', f'yl{i}'
            if xk not in d_data: break
            self.visuals.append((np.array(d_data[xk]).flatten(), np.array(d_data[yk]).flatten()))
            i += 1

        r_keys = sorted([k for k in c_data if k.startswith('r')], key=lambda x: int(x[1:]))
        for i in range(0, len(r_keys), 2):
            if i + 1 < len(r_keys):
                self.visuals.append((np.array(c_data[r_keys[i]]).flatten(), np.array(c_data[r_keys[i+1]]).flatten()))

class PhysicsBall:
    def __init__(self, level, start_idx=0):
        self.lvl = level
        self.idx = start_idx
        self.state = "ROLLING"
        self.lx = 0.01
        
        if self.idx in self.lvl.floors:
            self.ly = float(self.lvl.floors[self.idx]['func'](self.lx))
        else:
            self.idx = -1
            self.ly = 0.0
            self.state = "FALLING"

        self.lv = 0.0
        self.gx, self.gy = 0.0, 0.0
        self.gvx, self.gvy = 0.0, 0.0
        self.params = {'dt': 0.01, 'g': 9.81, 'fric': 0.01, 'rest': 0.3}

    def _to_global(self, lx, ly, ang):
        c, s = np.cos(ang), np.sin(ang)
        return lx * c - ly * s, lx * s + ly * c

    def _to_local(self, gx, gy, ang):
        c, s = np.cos(ang), np.sin(ang)
        return gx * c + gy * s, -gx * s + gy * c

    def update(self, angle, d_angle_dt):
        sub_steps = 10
        dt = self.params['dt'] / sub_steps
        g = self.params['g']

        for _ in range(sub_steps):
            if self.state == "ROLLING":
                if self.idx not in self.lvl.floors: self.state = "FALLING"; continue
                floor = self.lvl.floors[self.idx]
                slope = float(floor['slope'](self.lx))
                alpha = np.arctan(slope)
                
                accel = -g * np.sin(alpha + angle) - self.params['fric'] * self.lv
                self.lv += accel * dt
                self.lx += self.lv * dt
                self.ly = float(floor['func'](self.lx))
                
                self.gx, self.gy = self._to_global(self.lx, self.ly, angle)
                
                if self.lx < floor['min'] or self.lx > floor['max']:
                    self.state = "FALLING"
                    vx_loc = self.lv
                    vy_loc = slope * self.lv
                    
                    gvx_rel, gvy_rel = self._to_global(vx_loc, vy_loc, angle)
                    vx_tan = -d_angle_dt * self.gy
                    vy_tan = d_angle_dt * self.gx
                    
                    self.gvx, self.gvy = gvx_rel + vx_tan, gvy_rel + vy_tan

            elif self.state == "FALLING":
                self.gvy -= g * dt
                self.gx += self.gvx * dt
                self.gy += self.gvy * dt
                
                for f_idx, floor in self.lvl.floors.items():    
                    lx_chk, ly_chk = self._to_local(self.gx, self.gy, angle)
                    
                    if floor['min'] <= lx_chk <= floor['max']:
                        target_y = float(floor['func'](lx_chk))
                        
                        if ly_chk <= target_y:
                            slope = float(floor['slope'](lx_chk))
                            floor_angle = np.arctan(slope) + angle
                            
                            nx = -np.sin(floor_angle)
                            ny = np.cos(floor_angle)
                            
                            v_dot_n = self.gvx * nx + self.gvy * ny
                            
                            if v_dot_n < 0:
                                j = -(1 + self.params['rest']) * v_dot_n
                                self.gvx += j * nx
                                self.gvy += j * ny
                                
                                self.gx += nx * (target_y - ly_chk + 0.001)
                                self.gy += ny * (target_y - ly_chk + 0.001)

                                if abs(v_dot_n) < 0.5:
                                    self.state = "ROLLING"
                                    self.idx = f_idx
                                    self.lx = lx_chk
                                    self.ly = target_y
                                    
                                    tx = np.cos(floor_angle)
                                    ty = np.sin(floor_angle)
                                    self.lv = self.gvx * tx + self.gvy * ty
                                    break
        return self.gx, self.gy

class FuzzyController:
    def __init__(self):
        # Ordered keys for visualization purposes
        self.labels = ['NB', 'NS', 'Z', 'PS', 'PB']
        
        self.sets_pos = {
            'NB': -0.3, 'NS': -0.2, 'Z': 0.0, 'PS': 0.2, 'PB': 0.3
        }
        self.sets_vel = {
            'NB': -0.5, 'NS': -0.25, 'Z': 0.0, 'PS': 0.25, 'PB': 0.5
        }
        self.sets_out = {
            'NB': -0.5, 'NS': -0.25, 'Z': 0.0, 'PS': 0.25, 'PB': 0.5
        }
        
        # Rule Base stored as a list of tuples
        self.rules = [
            ('NB', 'NB', 'PB'), ('NB', 'NS', 'PB'), ('NB', 'Z',  'PB'), ('NB', 'PS', 'PB'), ('NB', 'PB', 'PS'), 
            ('NS', 'NB', 'PB'), ('NS', 'NS', 'PB'), ('NS', 'Z',  'PS'), ('NS', 'PS', 'PS'), ('NS', 'PB', 'Z'),
            ('Z',  'NB', 'PB'), ('Z',  'NS', 'PS'), ('Z',  'Z',  'NB'), ('Z',  'PS', 'NS'), ('Z',  'PB', 'NB'),
            ('PS', 'NB', 'Z'),  ('PS', 'NS', 'NS'), ('PS', 'Z',  'NS'), ('PS', 'PS', 'NB'), ('PS', 'PB', 'NB'),
            ('PB', 'NB', 'NS'), ('PB', 'NS', 'NB'), ('PB', 'Z',  'NB'), ('PB', 'PS', 'NB'), ('PB', 'PB', 'NB'),
        ]

    def _trimf(self, x, abc):
        a, b, c = abc
        # Triangular membership function
        return max(min((x - a) / (b - a + 1e-9), (c - x) / (c - b + 1e-9)), 0)

    def _get_memberships(self, val, sets):
        mems = {}
        keys = self.labels
        vals = [sets[k] for k in keys]
        
        for i, k in enumerate(keys):
            center = vals[i]
            # Determine neighbors for triangle width
            left = vals[i-1] if i > 0 else center - (vals[i+1]-center)
            right = vals[i+1] if i < len(vals)-1 else center + (center-vals[i-1])
            mems[k] = self._trimf(val, [left, center, right])
        return mems

    def compute(self, pos, vel):
        # Step 1: Fuzzification
        m_pos = self._get_memberships(pos, self.sets_pos)
        m_vel = self._get_memberships(vel, self.sets_vel)
        
        numerator = 0.0
        denominator = 0.0
        
        # Data for visualization
        active_rules = [] # Stores ((pos_idx, vel_idx), strength, out_val)
        
        # Step 2: Rule Evaluation
        for r_pos, r_vel, r_out in self.rules:
            # Min operator (AND)
            strength = min(m_pos[r_pos], m_vel[r_vel])
            
            if strength > 0:
                center = self.sets_out[r_out]
                numerator += strength * center
                denominator += strength
                
                # Store info for heatmap (indices for plotting)
                p_idx = self.labels.index(r_pos)
                v_idx = self.labels.index(r_vel)
                active_rules.append({'indices': (v_idx, p_idx), 'str': strength, 'out': center})
                
        # Step 3: Defuzzification (Center of Gravity / Weighted Average)
        if denominator == 0:
            out = 0.0
        else:
            out = numerator / denominator
            
        debug_info = {
            'm_pos': m_pos,
            'm_vel': m_vel,
            'rules': active_rules,
            'output': out
        }
        return out, debug_info

def run_controlled_simulation(level, ball):
    fuzzy = FuzzyController()
    
    # Pre-calculate static triangles for visualization background
    x_static = np.linspace(-1, 1, 100)
    static_plots = {'pos': [], 'vel': []}
    for k in fuzzy.labels:
        # Generate y-values for Position MF triangles
        vals = [fuzzy.sets_pos[x] for x in fuzzy.labels]
        i = fuzzy.labels.index(k)
        c = vals[i]
        l = vals[i-1] if i > 0 else c - (vals[i+1]-c)
        r = vals[i+1] if i < len(vals)-1 else c + (c-vals[i-1])
        y = [fuzzy._trimf(xi, [l, c, r]) for xi in x_static]
        static_plots['pos'].append((x_static, y, k))
        
        # Generate y-values for Velocity MF triangles
        vals = [fuzzy.sets_vel[x] for x in fuzzy.labels]
        i = fuzzy.labels.index(k)
        c = vals[i]
        l = vals[i-1] if i > 0 else c - (vals[i+1]-c)
        r = vals[i+1] if i < len(vals)-1 else c + (c-vals[i-1])
        y = [fuzzy._trimf(xi, [l, c, r]) for xi in x_static]
        static_plots['vel'].append((x_static, y, k))

    # History storage
    hist = {
        'x': [], 'y': [], 'angle': [], 
        'lx': [], 'lv': [], 'debug': [],
        'state': [] # 1 for rolling, 0 for falling
    }
    
    steps = 2000
    dt = 0.01
    current_angle = 0.0
    
    print("Simulating...")
    for s in range(steps):
        target_angle = 0.0
        debug_data = None
        
        if ball.state == "ROLLING":
            control_output, debug_data = fuzzy.compute(ball.lx, ball.lv)
            target_angle = -control_output 
            target_angle = np.clip(target_angle, -np.radians(45), np.radians(45))
        else:
            target_angle = 0.0
            # Empty debug data if falling
            debug_data = {'m_pos':{k:0 for k in fuzzy.labels}, 'm_vel':{k:0 for k in fuzzy.labels}, 'rules':[], 'output':0}

        angle_diff = target_angle - current_angle
        omega = np.clip(angle_diff / dt, -8.0, 8.0)
        new_angle = current_angle + omega * dt
        
        gx, gy = ball.update(new_angle, omega)
        current_angle = new_angle
        
        hist['x'].append(gx)
        hist['y'].append(gy)
        hist['angle'].append(current_angle)
        hist['lx'].append(ball.lx)
        hist['lv'].append(ball.lv)
        hist['state'].append(ball.state == "ROLLING")
        hist['debug'].append(debug_data)
        

    # --- Dashboard Setup ---
    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(3, 3)

    # 1. Main Track View (Top Left & Center)
    ax_track = fig.add_subplot(gs[0:2, 0:2])
    ax_track.set_title("Simulation")
    ax_track.set_xlim(-0.5, 0.5); ax_track.set_ylim(-0.5, 0.5); ax_track.set_aspect('equal')
    ax_track.grid(True, alpha=0.3)
    track_lines = [ax_track.plot([], [], 'k-', lw=1.5)[0] for _ in level.visuals]
    dot, = ax_track.plot([], [], 'ro', markersize=8, zorder=10)
    trace, = ax_track.plot([], [], 'r-', lw=0.5, alpha=0.5)

    # 2. Rule Heatmap (Top Right)
    ax_rules = fig.add_subplot(gs[0, 2])
    ax_rules.set_title("Active Rules Matrix")
    ax_rules.set_xticks(range(5)); ax_rules.set_xticklabels(fuzzy.labels)
    ax_rules.set_xlabel("Pos Error")
    ax_rules.set_yticks(range(5)); ax_rules.set_yticklabels(fuzzy.labels)
    ax_rules.set_ylabel("Velocity")
    # Initialize heatmap image (5x5 grid)
    rule_grid = np.zeros((5, 5)) 
    heatmap = ax_rules.imshow(rule_grid, cmap='Reds', vmin=0, vmax=1, origin='lower')

    # 3. Defuzzification View (Middle Right)
    ax_out = fig.add_subplot(gs[1, 2])
    ax_out.set_title("Defuzzification (Output)")
    ax_out.set_xlim(-0.6, 0.6); ax_out.set_ylim(0, 1.1)
    ax_out.grid(True, alpha=0.3)
    # Draw Singleton locations
    for k, v in fuzzy.sets_out.items():
        ax_out.axvline(v, color='gray', linestyle=':', alpha=0.5)
        ax_out.text(v, 1.02, k, ha='center', fontsize=8)
    # Dynamic bars for rule output strengths
    out_bars = ax_out.bar([v for v in fuzzy.sets_out.values()], [0]*5, width=0.05, color='blue', alpha=0.6)
    out_line = ax_out.axvline(0, color='red', lw=2, label='Result')
    
    # 4. Fuzzification Position (Bottom Left)
    ax_fuz_p = fig.add_subplot(gs[2, 0])
    ax_fuz_p.set_title("Fuzzification: Pos Error (lx)")
    ax_fuz_p.set_ylim(0, 1.1); ax_fuz_p.set_xlim(-0.4, 0.4)
    for x, y, k in static_plots['pos']:
        ax_fuz_p.plot(x, y, 'k-', lw=0.5, alpha=0.5)
        ax_fuz_p.fill_between(x, 0, y, alpha=0.05, color='blue')
        ax_fuz_p.text(fuzzy.sets_pos[k], 1.05, k, ha='center', fontsize=8)
    line_p_curr = ax_fuz_p.axvline(0, color='red', lw=1.5)
    dots_p, = ax_fuz_p.plot([], [], 'bo') # Intersection points

    # 5. Fuzzification Velocity (Bottom Center)
    ax_fuz_v = fig.add_subplot(gs[2, 1])
    ax_fuz_v.set_title("Fuzzification: Velocity (lv)")
    ax_fuz_v.set_ylim(0, 1.1); ax_fuz_v.set_xlim(-0.6, 0.6)
    for x, y, k in static_plots['vel']:
        ax_fuz_v.plot(x, y, 'k-', lw=0.5, alpha=0.5)
        ax_fuz_v.fill_between(x, 0, y, alpha=0.05, color='green')
        ax_fuz_v.text(fuzzy.sets_vel[k], 1.05, k, ha='center', fontsize=8)
    line_v_curr = ax_fuz_v.axvline(0, color='red', lw=1.5)
    dots_v, = ax_fuz_v.plot([], [], 'go') # Intersection points

    plt.tight_layout()

    def update(f):
        # -- Update Track --
        ang = hist['angle'][f]
        for ln, (lx, ly) in zip(track_lines, level.visuals):
            gx, gy = ball._to_global(lx, ly, ang)
            ln.set_data(gx, gy)
        dot.set_data([hist['x'][f]], [hist['y'][f]])
        trace.set_data(hist['x'][max(0, f-50):f], hist['y'][max(0, f-50):f])

        # -- Get Debug Data --
        info = hist['debug'][f]
        lx_val = hist['lx'][f]
        lv_val = hist['lv'][f]

        # -- Update Fuzzification Plots --
        line_p_curr.set_xdata([lx_val])
        line_v_curr.set_xdata([lv_val])
        
        # Calculate intersection y-values for dots
        p_dots_y = [info['m_pos'][k] for k in fuzzy.labels]
        p_dots_x = [fuzzy.sets_pos[k] for k in fuzzy.labels] # visual approximation (peaks)
        # Actually we want the dots on the red line, so x is always lx_val
        # But y is the membership value
        dots_p.set_data([lx_val]*5, p_dots_y)
        dots_v.set_data([lv_val]*5, [info['m_vel'][k] for k in fuzzy.labels])

        # -- Update Rule Heatmap --
        grid_data = np.zeros((5, 5))
        for r in info['rules']:
            # r['indices'] is (vel_idx, pos_idx) -> (row, col)
            row, col = r['indices']
            grid_data[row, col] = r['str']
        heatmap.set_data(grid_data)

        # -- Update Defuzzification --
        # Reset bars
        bar_heights = [0] * 5
        # Sum up strengths for each output singleton (max or sum depending on aggregation, here we visualize contribution)
        for r in info['rules']:
            # Find which singleton this rule maps to
            # We can find the index by matching the output value 'r['out']' to fuzzy.sets_out values
            out_val = r['out']
            # Find index in sets_out.values()
            vals = list(fuzzy.sets_out.values())
            try:
                idx = vals.index(out_val)
                # Accumulate for visualization (shows total pressure on that singleton)
                bar_heights[idx] = max(bar_heights[idx], r['str'])
            except ValueError: pass
            
        for bar, h in zip(out_bars, bar_heights):
            bar.set_height(h)
            
        out_line.set_xdata([info['output']])

        return track_lines + [dot, trace, line_p_curr, line_v_curr, dots_p, dots_v, heatmap, out_line] + list(out_bars)

    ani = FuncAnimation(fig, update, frames=len(hist['x']), interval=20, blit=False) # blit=False for Heatmap stability
    plt.show()

if __name__ == "__main__":
    lvl = Level('dificultad1.json', 'circulos.json')
    sim = PhysicsBall(lvl, start_idx=0)
    run_controlled_simulation(lvl, sim)