import os
import torch.optim as optim
import torch
import torch.nn as nn

class A2C_trainer:
    def __init__(self, actor_critic_policy, config):
        self.state_size = config['state_size']
        self.action_size = config['action_size']
        self.lr = config['learning_rate']
        self.model = actor_critic_policy
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.states, self.actions, self.rewards = [], [], []
        self.scores, self.episodes, self.average = [], [], []
        self.score = 0
        self.episode = 0
        self.EPISODES = config['number_of_episodes']
        self.max_average = -99999999999
        self.Save_Path = 'Models'
        if not os.path.exists(self.Save_Path):
            os.makedirs(self.Save_Path)
        self.Model_name = config['model_path']
        self.state = None
        self.time_of_day = None
        self.verbose = config['eplus_verbose']

    def update_global(self, global_a2c_agent):
        global_a2c_agent.update_global_net(local_model=self.model)
        global_a2c_agent.append_score(self.score)
        global_a2c_agent.evaluate_model()
        global_a2c_agent.save()

    def update_global_net(self, local_model):
        try:
            source_model = local_model
            target_model = self.model
            target_optimizer = self.optimizer

            with torch.no_grad():
                for source_param, target_param in zip(source_model.parameters(), target_model.parameters()):
                    if source_param.grad is not None:
                        if target_param.grad is None:
                            target_param.grad = torch.zeros_like(target_param)
                        target_param.grad.copy_(source_param.grad)

            target_optimizer.step()
            
        except Exception as e:
            error_message = f"An error occurred while updating the global network: {e}\n{traceback.format_exc()}"
            print(error_message)
            logging.error(error_message)

    def append_score(self, score):
        self.episode += 1
        self.episodes.append(self.episode)
        self.scores.append(score)
    
    def remember(self, state, action, reward):
        self.states.append(state)
        action_onehot = np.zeros([self.action_size])
        action_onehot[action] = 1
        self.actions.append(action_onehot)
        self.rewards.append(reward)
       
    def discount_rewards(self, reward):
        gamma = 0.99
        running_add = 0
        discounted_r = np.zeros_like(reward)
        for i in reversed(range(0, len(reward))):
            running_add = running_add * gamma + reward[i]
            discounted_r[i] = running_add
        discounted_r -= np.mean(discounted_r)
        discounted_r /= np.std(discounted_r)
        return discounted_r

    def replay(self):
        try:
            states = torch.FloatTensor(np.vstack(self.states))
            actions = torch.FloatTensor(np.vstack(self.actions))
            self.score = np.sum(self.rewards)
            discounted_r = torch.FloatTensor(self.discount_rewards(self.rewards))
            action_probs, values = self.model(states)
            values = values.squeeze()
            action_probs = torch.gather(action_probs, 1, actions.long())
            advantages = discounted_r - values
            actor_loss = -(torch.log(action_probs) * actions).sum(dim=1) * advantages
            critic_loss = advantages.pow(2)
            loss = actor_loss.mean() + critic_loss.mean()
            self.optimizer.zero_grad()
            loss.backward()
            if self.verbose == 0:
                pass
            elif self.verbose == 1:
                pass
            elif self.verbose == 2:
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        print(f"Gradients for {name}: {param.grad}")
            else:
                print("Warning: Verbose should be 0, 1 or 2 only")
            self.states, self.actions, self.rewards = [], [], []
        except Exception as e:
            error_message = f"An error occurred during the replay: {e}\n{traceback.format_exc()}"
            print(error_message)
            logging.error(error_message)
    
    def save(self, suffix=""):
        try:       
            torch.save(self.model.state_dict(), f"{self.Model_name[:-4]}{suffix}.pth")
        except Exception as e:
            error_message = f"Failed to save model: {e}\n{traceback.format_exc()}"
            print(error_message)
            logging.error(error_message)
    
    def evaluate_model(self):
        self.average.append(sum(self.scores[-50:]) / len(self.scores[-50:]))
        if __name__ == "__main__":
            if str(self.episode)[-1:] == "0":
                try:      
                    fig, ax = plt.subplots()              
                    ax.plot(self.episodes, self.scores, 'b')
                    ax.plot(self.episodes, self.average, 'r')
                    ax.set_ylabel('Score', fontsize=18)
                    ax.set_xlabel('Steps', fontsize=18)
                    ax.set_title("Episode scores")
                    fig.savefig(f"{self.Model_name[:-4]}.png")
                    plt.close('all')
                except OSError as e:
                    print(e)
                except:
                    e = sys.exc_info()[0]
                    print("Something else went wrong e: ", e)                                   
            if self.average[-1] >= self.max_average:
                self.max_average = self.average[-1]
                self.save(suffix="_best")
                SAVING = "SAVING"
            else:
                SAVING = ""
            print("episode: {}/{}, score: {}, average: {:.2f}, max average:{:.2f} {}".format(self.episode, self.EPISODES, self.scores[-1], self.average[-1],self.max_average, SAVING))

        return self.average[-1]


    ###### FROM INTERNET, FOR REFERENCE, DELETE Later ########

    def update(self, rollouts):
            obs_shape = rollouts.obs.size()[2:]
            action_shape = rollouts.actions.size()[-1]
            num_steps, num_processes, _ = rollouts.rewards.size()

            values, action_log_probs, dist_entropy, _ = self.actor_critic.evaluate_actions(
                rollouts.obs[:-1].view(-1, *obs_shape),
                rollouts.recurrent_hidden_states[0].view(
                    -1, self.actor_critic.recurrent_hidden_state_size),
                rollouts.masks[:-1].view(-1, 1),
                rollouts.actions.view(-1, action_shape))

            values = values.view(num_steps, num_processes, 1)
            action_log_probs = action_log_probs.view(num_steps, num_processes, 1)

            advantages = rollouts.returns[:-1] - values
            value_loss = advantages.pow(2).mean()

            action_loss = -(advantages.detach() * action_log_probs).mean()

            if self.acktr and self.optimizer.steps % self.optimizer.Ts == 0:
                # Compute fisher, see Martens 2014
                self.actor_critic.zero_grad()
                pg_fisher_loss = -action_log_probs.mean()

                value_noise = torch.randn(values.size())
                if values.is_cuda:
                    value_noise = value_noise.cuda()

                sample_values = values + value_noise
                vf_fisher_loss = -(values - sample_values.detach()).pow(2).mean()

                fisher_loss = pg_fisher_loss + vf_fisher_loss
                self.optimizer.acc_stats = True
                fisher_loss.backward(retain_graph=True)
                self.optimizer.acc_stats = False

            self.optimizer.zero_grad()
            (value_loss * self.value_loss_coef + action_loss -
            dist_entropy * self.entropy_coef).backward()

            if self.acktr == False:
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                        self.max_grad_norm)

            self.optimizer.step()

            return value_loss.item(), action_loss.item(), dist_entropy.item()