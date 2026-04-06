import subprocess

ppo_code = """
import torch
import torch.nn as nn
import numpy as np

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, obs):
        f = self.feature(obs)
        return self.actor_mean(f), torch.exp(self.actor_log_std), self.critic(f)
    
    def get_action_and_value(self, obs, action=None):
        mean, std, value = self.forward(obs)
        dist = torch.distributions.Normal(mean, std)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action).sum(-1), dist.entropy().sum(-1), value.squeeze(-1)


class PPOAgent:
    def __init__(self, obs_dim, action_dim, lr=3e-4, device='cpu'):
        self.device = torch.device(device)
        self.network = ActorCritic(obs_dim, action_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        self.policy_losses, self.value_losses, self.entropies = [], [], []
    
    def select_action(self, obs, deterministic=False):
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            mean, std, _ = self.network(obs_t)
            if deterministic:
                action = mean
            else:
                action = torch.distributions.Normal(mean, std).sample()
            log_prob = torch.distributions.Normal(mean, std).log_prob(action).sum(-1).item()
            value = self.network.critic(self.network.feature(obs_t)).squeeze(-1).item()
        return action.cpu().numpy()[0], log_prob, value
    
    def compute_gae(self, rewards, values, dones, next_value, gamma=0.99, lam=0.95):
        advantages = np.zeros_like(rewards)
        last_gae = 0
        for t in reversed(range(len(rewards))):
            next_val = next_value if t == len(rewards)-1 else values[t+1]
            delta = rewards[t] + gamma * next_val * (1-dones[t]) - values[t]
            last_gae = delta + gamma * lam * (1-dones[t]) * last_gae
            advantages[t] = last_gae
        return advantages, advantages + values
    
    def update(self, rollout_data, epochs=4, eps=0.2, vf_coef=0.5, ent_coef=0.01):
        obs, actions, old_log_probs, advantages, returns = rollout_data
        obs = torch.FloatTensor(obs).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        for _ in range(epochs):
            _, new_log_probs, entropy, values = self.network.get_action_and_value(obs, actions)
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1-eps, 1+eps) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = 0.5 * ((returns - values) ** 2).mean()
            loss = policy_loss + vf_coef * value_loss - ent_coef * entropy.mean()
            
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
            self.optimizer.step()
            
            self.policy_losses.append(policy_loss.item())
            self.value_losses.append(value_loss.item())
            self.entropies.append(entropy.mean().item())


if __name__ == '__main__':
    agent = PPOAgent(51, 15)
    print('PPO Agent created successfully!')
    print(f'Parameters: {sum(p.numel() for p in agent.network.parameters())}')
"""

with open(
    "/mnt/stuff-first/Projects/SpiderLegFingers/spider_leg_fingers/src/rl/agents/ppo.py",
    "w",
) as f:
    f.write(ppo_code.strip())

print("Created ppo.py")
