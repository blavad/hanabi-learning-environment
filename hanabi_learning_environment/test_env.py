import unittest
from hanabi_learning_environment.rl_env import HanabiIntentEnv, HanabiEnv

class HanabiIntentTest(unittest.TestCase):
    
    def test_massive_discard(self):
        configs = { "players":2,
                    "intent_type":["pdn","pdn"]}

        env = HanabiIntentEnv(configs)
        _ = env.reset()
        _, _, _, _ = env.step(1)
        _, _, _, _ = env.step(1)
        _, _, _, _ = env.step(1)
        _, _, _, _ = env.step(1)
        _, _, _, _ = env.step(1)
        _, _, _, _ = env.step(1)
        _, _, _, _ = env.step(1)


    def test_massive_play(self):
        configs = { "players":2,
                    "intent_type":["pdn","pdn"]}

        env = HanabiIntentEnv(configs)
        _ = env.reset()
        _, _, _, _ = env.step(0)
        _, _, _, _ = env.step(0)
        _, _, _, _ = env.step(0)
        _, _, _, _ = env.step(0)
        _, _, _, _ = env.step(0)
        _, _, _, _ = env.step(0)
        _, _, _, _ = env.step(0)
        


    
class HanabiTest(unittest.TestCase):
    
    def test_init_players(self):
        self.assertEqual(HanabiEnv({ "players" : 2}).players, 2)
        self.assertEqual(HanabiEnv({ "players" : 3}).players, 3)
        self.assertEqual(HanabiEnv({ "players" : 4}).players, 4)
        self.assertEqual(HanabiEnv({ "players" : 5}).players, 5)
