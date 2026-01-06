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
        self.sets_pos = {
            'NB': -0.3, 'NS': -0.1, 'Z': 0.0, 'PS': 0.1, 'PB': 0.3
        }
        self.sets_vel = {
            'NB': -0.5, 'NS': -0.25, 'Z': 0.0, 'PS': 0.25, 'PB': 0.5
        }
        self.sets_out = {
            'NB': -0.5, 'NS': -0.25, 'Z': 0.0, 'PS': 0.25, 'PB': 0.5
        }
        
        self.rules = [
            ('NB', 'NB', 'PB'),
            ('NB', 'NS', 'PB'),
            ('NB', 'Z',  'PB'),
            ('NB', 'PS', 'PB'), 
            ('NB', 'PB', 'PS'), 

            ('NS', 'NB', 'PB'),
            ('NS', 'NS', 'PB'),
            ('NS', 'Z',  'PS'),
            ('NS', 'PS', 'PS'),
            ('NS', 'PB', 'Z'),

            ('Z',  'NB', 'PB'),
            ('Z',  'NS', 'PS'),
            ('Z',  'Z',  'NB'),
            ('Z',  'PS', 'NS'),
            ('Z',  'PB', 'NB'),

            ('PS', 'NB', 'Z'),
            ('PS', 'NS', 'NS'),
            ('PS', 'Z',  'NS'),
            ('PS', 'PS', 'NB'),
            ('PS', 'PB', 'NB'),

            ('PB', 'NB', 'NS'),
            ('PB', 'NS', 'NB'),
            ('PB', 'Z',  'NB'),
            ('PB', 'PS', 'NB'),
            ('PB', 'PB', 'NB'),
        ]

    def _trimf(self, x, abc):
        a, b, c = abc
        return max(min((x - a) / (b - a + 1e-9), (c - x) / (c - b + 1e-9)), 0)

    def _get_memberships(self, val, sets):
        mems = {}
        keys = list(sets.keys())
        vals = list(sets.values())
        
        for i, k in enumerate(keys):
            center = vals[i]
            left = vals[i-1] if i > 0 else center - (vals[i+1]-center)
            right = vals[i+1] if i < len(vals)-1 else center + (center-vals[i-1])
            mems[k] = self._trimf(val, [left, center, right])
        return mems

    def compute(self, pos, vel):
        m_pos = self._get_memberships(pos, self.sets_pos)
        m_vel = self._get_memberships(vel, self.sets_vel)
        
        numerator = 0.0
        denominator = 0.0
        
        for r_pos, r_vel, r_out in self.rules:
            strength = min(m_pos[r_pos], m_vel[r_vel])
            if strength > 0:
                center = self.sets_out[r_out]
                numerator += strength * center
                denominator += strength
                
        if denominator == 0:
            return 0.0
        return numerator / denominator

def run_controlled_simulation(level, ball):
    fuzzy = FuzzyController()
    
    hist = {'x': [], 'y': [], 'angle': [], 'target': []}
    steps = 2000
    dt = 0.01
    current_angle = 0.0
    
    print("Simulating...")
    for s in range(steps):
        target_angle = 0.0
        
        if ball.state == "ROLLING":
            control_output = fuzzy.compute(ball.lx, ball.lv)
            target_angle = -control_output 
            
            max_ang = np.radians(45)
            target_angle = np.clip(target_angle, -max_ang, max_ang)
        else:
            target_angle = 0.0

        angle_diff = target_angle - current_angle
        omega = angle_diff / dt 
        
        max_omega = 8.0 
        omega = np.clip(omega, -max_omega, max_omega)
        
        new_angle = current_angle + omega * dt
        
        gx, gy = ball.update(new_angle, omega)
        current_angle = new_angle
        
        hist['x'].append(gx)
        hist['y'].append(gy)
        hist['angle'].append(current_angle)
        
        tx, ty = ball._to_global(0, ball.ly if ball.state=="ROLLING" else 0, current_angle)
        hist['target'].append((tx, ty))
        
        if gy < -1.5:
            print("Ball reached bottom.")
            break

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 0.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title("Fuzzy Controller")

    lines = [ax.plot([], [], 'k-', lw=1.5)[0] for _ in level.visuals]
    dot, = ax.plot([], [], 'ro', markersize=8, zorder=10)
    target_mark, = ax.plot([], [], 'g+', markersize=10, markeredgewidth=2)
    trace, = ax.plot([], [], 'r-', lw=0.5, alpha=0.5)

    def anim(f):
        a = hist['angle'][f]
        for ln, (lx, ly) in zip(lines, level.visuals):
            gx, gy = ball._to_global(lx, ly, a)
            ln.set_data(gx, gy)
        
        dot.set_data([hist['x'][f]], [hist['y'][f]])        
        trace.set_data(hist['x'][max(0, f-50):f], hist['y'][max(0, f-50):f])
        
        tx, ty = hist['target'][f]
        target_mark.set_data([tx], [ty])
        
        return lines + [dot, trace, target_mark]

    ani = FuncAnimation(fig, anim, frames=len(hist['x']), interval=10, blit=True)
    plt.show()

if __name__ == "__main__":
    lvl = Level('dificultad1.json', 'circulos.json')
    sim = PhysicsBall(lvl, start_idx=0)
    run_controlled_simulation(lvl, sim)