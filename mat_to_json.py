import json
import numpy as np
import scipy.io as sio

def save_mat_to_json(mat_vars, filename):
    # Clean the dictionary: convert numpy arrays to lists and remove metadata
    cleaned_data = {}
    for key, value in mat_vars.items():
        if not key.startswith('__'):
            # Convert numpy arrays to lists for JSON compatibility
            if isinstance(value, np.ndarray):
                cleaned_data[key] = value.tolist()
            else:
                cleaned_data[key] = value

    with open(filename, 'w') as f:
        json.dump(cleaned_data, f, indent=4)
        
if __name__ == "__main__":
    mat_diff = sio.loadmat('dificultad1.mat')
    mat_dif2 = sio.loadmat('dificultad2.mat')
    mat_dif3 = sio.loadmat('dificultad3.mat')
    mat_dif4 = sio.loadmat('dificultad4.mat')
    mat_circ = sio.loadmat('circulos.mat')
    save_mat_to_json(mat_diff, 'dificultad1.json')
    save_mat_to_json(mat_dif2, 'dificultad2.json')
    save_mat_to_json(mat_dif3, 'dificultad3.json')
    save_mat_to_json(mat_dif4, 'dificultad4.json')
    save_mat_to_json(mat_circ, 'circulos.json')