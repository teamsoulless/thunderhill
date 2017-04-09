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


def process_output(infile, outfile, base_dir, correction=0.0, interpolate_acc=False, interpolate_brake=False,
                   scale_steering=True):
    data = pd.read_csv(os.path.join(base_dir, infile), header=0)

    velocities = data[['vel0', 'vel1']].as_matrix()

    if interpolate_acc:
        acc = accel_from_velocity(velocities)

        nan_idx = np.where(np.isnan(acc))
        acc[nan_idx[0]] = 0

        acc[acc < 0] = 0
        acc[data['brake'] > 0.02] = 0
        acc = (acc - np.min(acc)) / (np.max(acc) - np.min(acc))
        data['throttle'] = acc

        data['brake'] -= correction
        data['brake'][data['brake'] < 0] = 0
        data['brake'] = (data['brake'] - np.min(data['brake'])) / (np.max(data['brake']) - np.min(data['brake']))

    if interpolate_brake:
        acc = accel_from_velocity(velocities)

        nan_idx = np.where(np.isnan(acc))
        acc[nan_idx[0]] = 0

        acc *= -1
        acc[acc < 0] = 0
        acc = (acc - np.min(acc)) / (np.max(acc) - np.min(acc))
        data['brake'] = acc

        # data['throttle'] -= correction
        data['throttle'][np.array(data['throttle']) < 0] = 0
        data['brake'][np.array(data['throttle']) > 1e-5] = 0
        # data['throttle'] = data['throttle'].apply(lambda x: 0 if x < 0 else x)
        # data['brake'] = data['throttle'].apply(lambda x: 0 if x > 0 else x)
        data['throttle'] = (data['throttle'] - np.min(data['throttle'])) \
                           / (np.max(data['throttle']) - np.min(data['throttle']))

    if scale_steering:
        data['steering'] *= -10.

    # data = data[:3500]
    data.to_csv(os.path.join(base_dir, outfile), index=False)
    return data


def load_and_visualize(dir, infile):
    data = pd.read_csv(os.path.join(dir, infile), header=0)

    plot = plt.plot(data['steering'])
    # plt.imsave(os.path.join(dir, 'steering.png'), plot)


# load_and_visualize('D:\\DAY1_raw\\1050', 'outputNew.txt')

if __name__ == '__main__':
    base_dir = 'D:\\DAY2_raw'
    process_dirs = [
        '1050'
    ]

    INFILE = 'outputNew.txt'
    OUTFILE = 'output_processed.txt'

    for file in process_dirs:
        data = process_output(
            infile=INFILE,
            outfile=OUTFILE,
            base_dir=os.path.join(base_dir, file),
            correction=0.035,
            interpolate_acc=False,
            interpolate_brake=True,
            scale_steering=False
          )
        plt.plot(data['steering'])
        plt.plot(data['throttle'])
        plt.plot(data['brake'])
        plt.legend(loc='center left', bbox_to_anchor=(0, 0.1))
        # plt.tight_layout()
        plt.savefig(os.path.join(base_dir, file, 'fig.png'))
        # plt.close()
