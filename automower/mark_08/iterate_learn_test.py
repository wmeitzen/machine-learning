
from simple_lawnmower_sjhatfield.qlearning import QLearning
from simple_lawnmower_sjhatfield.agent_testing import AgentTesting
from simple_lawnmower_sjhatfield.env import BlobEnv

env = BlobEnv()
qlearning = QLearning(env = env)
agent_testing = AgentTesting(env = env)

agent_testing.set_number_of_runs(5) # Number of times to draw the mower and lawn; has no bearing on learning
agent_testing.set_render_waitkey(20)  # 0 draws an empty lawn, but is as fast as possible
agent_testing.set_debug_render(True) # True: show the lawn; False: do not show the lawn

# - experimenting with lawn_size 8
# ok to delete all lawn_size 8s
lawn_size_values = [5, 7, 8, 10, 12, 15, 18] # best: 10
epsilon_decay_values = [0.99, 0.995, 0.999] # best: 0.99
learning_rate_decay_values = [0.999, 0.9999, 0.99999, 0.999999, 0.9999999] # best: 0.99999
episode_length_values = [1_000, 2_000, 3_000] # best: 2000
discount_values = [0.99, 0.999] # best: 0.99
for lawn_size in lawn_size_values:
    qlearning.set_size(lawn_size)
    qlearning.set_lawn_name(f"open")
    for epsilon_decay in epsilon_decay_values:
        qlearning.set_epsilon_decay(epsilon_decay)
        for learning_rate_decay in learning_rate_decay_values:
            qlearning.set_learning_rate_decay(learning_rate_decay)
            for episode_length in episode_length_values:
                qlearning.set_num_episodes(episode_length)
                for discount in discount_values:
                    print(f"\n-------- New Run --------")
                    print(f"lawn_name: {qlearning.get_lawn_name()}")
                    print(f"lawn_size: {lawn_size}")
                    print(f"epsilon_decay: {epsilon_decay}")
                    print(f"learning_rate_decay: {learning_rate_decay}")
                    print(f"episode_length: {episode_length}")
                    print(f"discount: {discount}")
                    qlearning.set_discount(discount)
                    qlearning.launch_qlearning()
                    db_trial = qlearning.get_db_trial()
                    agent_testing.launch_agent_testing(db_trial = db_trial)
