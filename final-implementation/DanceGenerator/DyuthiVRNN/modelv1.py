import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import imageio

class PoseVRNN(nn.Module):
    def __init__(self, pose_dim, h_dim, z_dim, n_layers):
        super(PoseVRNN, self).__init__()

        self.pose_dim = pose_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.n_layers = n_layers

        # Feature extractors
        self.phi_x = nn.Sequential(
            nn.Linear(pose_dim, h_dim),
            nn.ReLU()
        )
        self.phi_z = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.ReLU()
        )

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(h_dim + h_dim, h_dim),
            nn.ReLU()
        )
        self.encoder_mean = nn.Linear(h_dim, z_dim)
        self.encoder_std = nn.Sequential(
            nn.Linear(h_dim, z_dim),
            nn.Softplus()
        )

        # Prior
        self.prior = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU()
        )
        self.prior_mean = nn.Linear(h_dim, z_dim)
        self.prior_std = nn.Sequential(
            nn.Linear(h_dim, z_dim),
            nn.Softplus()
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(h_dim + h_dim, h_dim),
            nn.ReLU()
        )
        self.decoder_mean = nn.Linear(h_dim, pose_dim)

        # RNN
        self.rnn = nn.GRU(h_dim + h_dim, h_dim, n_layers)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        h = torch.zeros(self.n_layers, batch_size, self.h_dim, device=x.device)

        kld_loss, mse_loss = 0, 0
        all_predictions = []

        for t in range(seq_len - 1):
            x_t = x[:, t, :]
            x_t_next = x[:, t + 1, :]

            # Feature extraction
            phi_x_t = self.phi_x(x_t)

            # Encoder
            enc_input = torch.cat([phi_x_t, h[-1]], dim=-1)
            enc_t = self.encoder(enc_input)
            enc_mean_t = self.encoder_mean(enc_t)
            enc_std_t = self.encoder_std(enc_t)

            # Prior
            prior_t = self.prior(h[-1])
            prior_mean_t = self.prior_mean(prior_t)
            prior_std_t = self.prior_std(prior_t)

            # Latent sampling
            z_t = self._reparameterized_sample(enc_mean_t, enc_std_t)
            phi_z_t = self.phi_z(z_t)

            # Decoder
            dec_input = torch.cat([phi_z_t, h[-1]], dim=-1)
            dec_t = self.decoder(dec_input)
            x_t_pred = self.decoder_mean(dec_t)

            # Update RNN state
            rnn_input = torch.cat([phi_x_t, phi_z_t], dim=-1).unsqueeze(0)
            _, h = self.rnn(rnn_input, h)

            # Losses
            kld_loss += self._kld_gauss(enc_mean_t, enc_std_t, prior_mean_t, prior_std_t)
            mse_loss += torch.mean((x_t_next - x_t_pred) ** 2)

            all_predictions.append(x_t_pred.unsqueeze(1))

        return kld_loss, mse_loss, torch.cat(all_predictions, dim=1)

    def _reparameterized_sample(self, mean, std):
        eps = torch.randn_like(std)
        return mean + eps * std

    def _kld_gauss(self, mean_1, std_1, mean_2, std_2):
        return torch.sum(0.5 * (2 * torch.log(std_2 / std_1) +
                                (std_1 ** 2 + (mean_1 - mean_2) ** 2) / (std_2 ** 2) - 1))

pose_dim = 4  # each pose ; 8 earlier
h_dim = 64    # RNN hidden state
z_dim = 64    # latent space
n_layers = 2  # RNN layers

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'mps' if torch.backends.mps.is_available() and torch.backends.mps.is_built() else device

device = torch.device(device)


def draw_stick_figure(ax, left_shoulder_angle = 30, right_shoulder_angle = 30, left_elbow_angle = 170, right_elbow_angle = 170, left_hip_angle = 170, right_hip_angle = 170, left_knee_angle = 170, right_knee_angle = 170):
    upper_arm_length = 1.0
    forearm_length = 1.0
    thigh_length = 1.0
    calf_length = 1.0
    head_radius = 0.3


    left_shoulder_rad = np.radians(left_shoulder_angle)
    right_shoulder_rad = np.radians(right_shoulder_angle)

    left_hip_angle = 180 - left_hip_angle
    right_hip_angle = 180 - right_hip_angle

    left_hip_rad = np.radians(left_hip_angle)
    right_hip_rad = np.radians(right_hip_angle)


    # Torso - fixed points
    torso_top = np.array([0, 2])
    torso_bottom = np.array([0, 0])
    head_center = torso_top + np.array([0, head_radius + 0.1])

    # Right arm
    right_shoulder = torso_top
    right_elbow = right_shoulder + upper_arm_length * np.array([np.sin(right_shoulder_rad), -np.cos(right_shoulder_rad)])
    right_hand = right_elbow + forearm_length * np.array([-np.cos(np.radians(90 - right_shoulder_angle + right_elbow_angle)), np.sin(np.radians(90 - right_shoulder_angle + right_elbow_angle))])
    # print("shape:", right_elbow.shape)
    # print("shape 2:", right_hand.shape)

    # Left arm
    left_shoulder = torso_top
    left_elbow = left_shoulder + upper_arm_length * np.array([-np.sin(left_shoulder_rad), -np.cos(left_shoulder_rad)])
    # print("left_shoulder_angle:", left_shoulder_angle.shape, left_shoulder_angle)
    left_hand = left_elbow + forearm_length * np.array([np.cos(np.radians(90 - left_shoulder_angle + left_elbow_angle)), np.sin(np.radians(90 - left_shoulder_angle + left_elbow_angle))])

    # Right leg
    right_hip = torso_bottom
    right_knee = right_hip + thigh_length * np.array([np.sin(right_hip_rad), -np.cos(right_hip_rad)])
    right_foot = right_knee + calf_length * np.array([-np.cos(np.radians(right_knee_angle + right_hip_angle - 90)), -np.sin(np.radians(right_knee_angle + right_hip_angle - 90))])

    # Left leg
    left_hip = torso_bottom
    left_knee = left_hip + thigh_length * np.array([-np.sin(left_hip_rad), -np.cos(left_hip_rad)])
    left_foot = left_knee + calf_length * np.array([np.cos(np.radians(left_knee_angle + left_hip_angle - 90)), -np.sin(np.radians(left_knee_angle + left_hip_angle - 90))])

    ax.plot([torso_bottom[0], torso_top[0]], [torso_bottom[1], torso_top[1]], 'k-', lw=4)  # Torso

    # Right arm
    ax.plot([right_shoulder[0], right_elbow[0]], [right_shoulder[1], right_elbow[1]], 'r-', lw=4)
    ax.plot([right_elbow[0], right_hand[0]], [right_elbow[1], right_hand[1]], 'r-', lw=4)

    # Left arm
    ax.plot([left_shoulder[0], left_elbow[0]], [left_shoulder[1], left_elbow[1]], 'r-', lw=4)
    ax.plot([left_elbow[0], left_hand[0]], [left_elbow[1], left_hand[1]], 'r-', lw=4)

    # Right leg
    ax.plot([right_hip[0], right_knee[0]], [right_hip[1], right_knee[1]], 'b-', lw=4)
    ax.plot([right_knee[0], right_foot[0]], [right_knee[1], right_foot[1]], 'b-', lw=4)

    # Left leg
    ax.plot([left_hip[0], left_knee[0]], [left_hip[1], left_knee[1]], 'b-', lw=4)
    ax.plot([left_knee[0], left_foot[0]], [left_knee[1], left_foot[1]], 'b-', lw=4)

    # Head as a circle
    head = plt.Circle(head_center, head_radius, color='k', fill=False, lw=2)
    ax.add_patch(head)

    ax.set_xlim(-3, 3)
    ax.set_ylim(-2, 3)
    ax.set_aspect('equal')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')

    # plt.show()



def get_model(state_dict_path="/Users/soardr/PycharmProjects/ReinforcementLearningSnakeGame/DanceGenerator/models/vrnn_trial_2__only_top__epoch14.pth"):
    vrnn_model = PoseVRNN(pose_dim, h_dim, z_dim, n_layers).to(device)

    # Map to MPS backend
    state_dict_mps = torch.load(state_dict_path, map_location=torch.device("mps"))

    vrnn_model.load_state_dict(state_dict_mps)

    # Move model to MPS
    vrnn_model.to(device)

    return vrnn_model

def generate_sequence(vrnn_model: PoseVRNN, initial_pose_original: torch.Tensor = None):
    print("inside generate_sequence function...")
    initial_pose = initial_pose_original.clone().detach()
    print("generate_sequence pointer 1")

    vrnn_model.eval()
    print("generate_sequence pointer 2")

    with torch.no_grad():
        print("generate_sequence pointer 3")
        seq_len = 60
        generated_sequence = []

        initial_pose = initial_pose.to(device)
        print("generate_sequence pointer 4")

        try:
            h = torch.zeros(n_layers, 1, h_dim, device=device)
            for _ in range(seq_len):
                # print("generate_sequence pointer 4.5")
                try:
                    # print("initial_pose:", initial_pose.shape, initial_pose)
                    
                    initial_pose_temp = initial_pose.unsqueeze(0)
                    # initial_pose_temp = initial_pose
                    
                    phi_x = vrnn_model.phi_x(initial_pose_temp)
                except Exception as e_phi_x:
                    print("e_phi_x:", e_phi_x)
                # print("phi_x:", phi_x)

                try:
                    prior = vrnn_model.prior(h[-1])
                except Exception as e_prior:
                    print("e_prior:", e_prior)
                # print("prior:", type(prior))

                try:
                    prior_mean = vrnn_model.prior_mean(prior)
                except Exception as e_prior_mean:
                    print("e_prior_mean:", e_prior_mean)
                # print("prior_mean:", type(prior_mean))

                try:
                    prior_std = vrnn_model.prior_std(prior)
                except Exception as e_prior_std:
                    print("e_prior_std:", e_prior_std)
                # print("prior_std:", type(prior_std))

                try:
                    z = vrnn_model._reparameterized_sample(prior_mean, prior_std)
                except Exception as e_z:
                    print("e_z:", e_z)
                # print("generate_sequence pointer 5")

                try:
                    phi_z = vrnn_model.phi_z(z)
                except Exception as e_phi_z:
                    print("e_phi_z:", e_phi_z)
                # print("phi_z:", phi_z)

                dec_input = torch.cat([phi_z, h[-1]], dim=-1)
                dec_t = vrnn_model.decoder(dec_input)
                # print("generate_sequence pointer 6")

                next_pose = vrnn_model.decoder_mean(dec_t)
                # print("generate_sequence pointer 7")

                generated_sequence.append(next_pose.cpu().numpy())
                initial_pose = next_pose  # feed the output back as input

                # print("shapes of phi_x, phi_z:", phi_x.shape, phi_z.shape)
                
                phi_x_reshaped = phi_x.reshape(phi_z.shape)
                # print("shapes of phi_x_reshaped, phi_z:", phi_x_reshaped.shape, phi_z.shape)  

                rnn_input = torch.cat([phi_x_reshaped, phi_z], dim=-1).unsqueeze(0)
                _, h = vrnn_model.rnn(rnn_input, h)
        except Exception as e:
            print("error in generate_sequence un modelv1.py:", e)

    generated_sequence = np.array(generated_sequence)

    return generated_sequence


def create_animation(angle_sequences, output_path="stick_figure.gif", save_animation=False, num_reps=10):
    temp_dir = "temp_frames"
    if save_animation is True:
        os.makedirs(temp_dir, exist_ok=True)
    filenames = []

    print("inside create_animation function:", angle_sequences.shape, angle_sequences)

    try:
        for _ in range(num_reps):
            for i, angles in enumerate(angle_sequences):
                print("values:", angles.shape, angles)

                fig, ax = plt.subplots()
                # draw_stick_figure(ax, left_elbow_angle=0, left_shoulder_angle=angles[0][0], left_knee_angle=0,
                #     right_elbow_angle=angles[0][1], right_shoulder_angle=0, right_knee_angle=angles[0][2],
                #     left_hip_angle=0, right_hip_angle=angles[0][3])

                draw_stick_figure(ax, right_elbow_angle=angles[0][0], right_shoulder_angle=angles[0][1],
                                left_shoulder_angle=angles[0][2], left_elbow_angle=angles[0][3])
                
                if save_animation is True:
                    filename = f"{temp_dir}/frame_{i:03d}.png"
                    filenames.append(filename)
                    plt.savefig(filename)
                    plt.close(fig)
    except Exception as e_create_animation_1:
        print("error in create_animation pointer 1:", e_create_animation_1)
    
    if save_animation is True:
        try:
            with imageio.get_writer(output_path, mode="I", duration=0.2, loop=0) as writer:
                for filename in filenames:
                    image = imageio.imread(filename)
                    writer.append_data(image)
        except Exception as e_create_animation_2:
            print("error in create_animation pointer 2:", e_create_animation_2)
        
        for filename in filenames:
            os.remove(filename)
        os.rmdir(temp_dir)
