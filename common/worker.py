from .atari_wrappers import make_atari
from multiprocessing import Process


class Worker(Process):
    def __init__(self, id, conn, **config):
        super(Worker, self).__init__()
        self.id = id
        self.config = config
        self.env = make_atari(self.config["env_name"], episodic_life=False, seed=self.config["seed"] + self.id)
        self.conn = conn
        self.reward = 0
        self.episode_buffer = []

    def __str__(self):
        return str(self.id)

    def render(self):
        self.env.render()

    def run(self):
        print(f"W{self.id}: started.")
        state = self.env.reset()
        while True:
            self.conn.send(state)
            action, value = self.conn.recv()
            next_state, reward, done, info = self.env.step(action)
            # self.render()
            self.conn.send((next_state, reward, done))
            self.episode_buffer.append((state, action, reward, done, value))
            state = next_state
            if done:
                self.conn.send(self.episode_buffer)
                self.episode_buffer = []
                state = self.env.reset()