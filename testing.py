# from gym_minigrid.wrappers import *
# env = gym.make('MiniGrid-Empty-5x5-v0')
# env = RGBImgPartialObsWrapper(env) # Get pixel observations
# env = ImgObsWrapper(env) # Get rid of the 'mission' field
# obs = env.reset() # This now produces an RGB tensor only
#
# for i_episode in range(2):
#     observation = env.reset()
#     for t in range(10):
#         env.render()
#         print(observation)
#         action = env.action_space.sample()
#         observation, reward, done, info = env.step(action)
#         if done:
#             print("Episode finished after {} timesteps".format(t+1))
#             break
# env.close()


a = [0,0,0,1,1]
print(list(enumerate(a)))