import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def accel_from_velocity(velocities):
    n_obs = velocities.shape[0]
    acceleration_vectors = velocities[1:] - velocities[:(n_obs-1)]

    velocity_magnitudes = []
    for v1, v2 in zip(velocities[:, 0], velocities[:, 1]):
        if abs(v1) < 1e-5 and abs(v2) < 1e-5:
            velocity_magnitudes.append(np.inf)
        else:
            velocity_magnitudes.append(np.sqrt(v1**2 + v2**2))

    velocity_magnitudes = np.array(velocity_magnitudes)

    normalized_velocities = velocities[1:] / velocity_magnitudes[1:].reshape(-1, 1)

    normalized_acceleration = np.array([normalized_velocities[k, :] @ acceleration_vectors[k, :] for k in range(n_obs-1)])
    normalized_acceleration[normalized_acceleration > 1] = 0

    return np.insert(normalized_acceleration, 0, [0], axis=0)


def process_output(infile, outfile, base_dir, interpolate_acc=True, scale_steering=True):
    data = pd.read_csv(os.path.join(base_dir, infile), header=0)

    velocities = data[['vel0', 'vel1']].as_matrix()

    if interpolate_acc:
        data['accel'] = accel_from_velocity(velocities)
    if scale_steering:
        data['steering'] *= -10.

    data.to_csv(base_dir + outfile, index=False)


if __name__ == '__main__':
    base_dir = 'D:\\'
    process_dirs = []

    INFILE = 'output.txt'
    OUTFILE = 'output_processed.txt'

    for file in process_dirs:
        process_output(
            infile=INFILE,
            outfile=OUTFILE,
            base_dir=os.path.join(base_dir, file),
            interpolate_acc=True,
            scale_steering=True
          )
