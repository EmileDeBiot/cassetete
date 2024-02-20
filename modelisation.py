import numpy as np
import matplotlib.pyplot as plt

import torch
import tqdm
import itertools

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

piece = np.array([[0,0,0],[1,0,0],[2,0,0],[3,0,0],[2,1,0]])
piece_max = np.array([[4,4,4],[0,0,0],[0,4,4],[4,0,4],[4,4,0]])

def init_random_pieces():
    """Generate initial random pieces.
    
    returns:
        list[Piece]"""
    all_coord = []
    for i in range(5):
        for j in range(5):
            for k in range(5):
                all_coord.append([i,j,k])
    np.random.shuffle(all_coord)
    pieces = []
    for i in range(0,len(all_coord),5):
        piece = np.array([all_coord[i+k] for k in range(5)])
        pieces.append(Piece(piece))
    return pieces

def is_valid(piece):
    """Check if a piece is valid.
    
    args: 
        piece: np.array
    
    returns:
        bool"""
    in_plan = False
    plan = 0
    for j in range(3):
        if np.all([piece[i,j] == piece[i+1,j] for i in range(4)]):
            in_plan = True
            plan = j
            break
    if not in_plan:
        return False
    for c in range(5):
        for j in range(3):
            if j != plan and sum(piece[i,j] == c for i in range(5)) == 4:
                return True
    return False

class Piece:
    piece : np.array
    loss : float
    def __init__(self, piece):
        self.piece = piece
        self.update_loss()

    def update_loss(self):
        """Update the loss of the piece."""
        somme = 0
        for i in range(len(self.piece)):
            for j in range(i+1, len(self.piece)):
                somme += sum(np.abs(self.piece[i][k] - self.piece[j][k]) for k in range(3))
        plan = False
        for j in range(3):
            if np.all([self.piece[i,j] == self.piece[i+1,j] for i in range(4)]):
                plan = True
                break
        self.loss = (1-np.abs((somme-18)/54))**2 # la puissance est là pour que la loss soit plus sensible aux petites variations prohe de 1.
        if plan and is_valid(self.piece): # il faut que ce soit petit sinon les pièces sont trop rigides.
            self.loss *= 1.005
        if is_valid(self.piece):
            self.loss *= 1.01
    
    def show(self):
        print(self.piece)

def indice(i):
    """Get the indices of a piece from its index in the cube."""
    return int(i//5),int(i%5)

class Modelisation:
    cube_cible = np.zeros((5,5,5))
    pieces : list[Piece]
    def __init__(self):
        self.cube = np.zeros((5,5,5))
        self.pieces = init_random_pieces()
        for i in range(len(self.pieces)):
            for j in range(5):
                self.cube_cible[self.pieces[i].piece[j][0],self.pieces[i].piece[j][1],self.pieces[i].piece[j][2]] = j + 5*i 
    
    def loss(self):
        """Get the loss of the modelisation."""
        return np.sum([piece.loss for piece in self.pieces])/len(self.pieces)
    
    def step(self, epsilon = 0.001, zeta = 0.0001, nu = 0.001):
        """Make a step of the modelisation."""
        l_current = self.loss()
        min_piece = self.pieces[0].loss
        min_piece_i_1 = 0
        min_piece_i_2 = 1
        for i in range(len(self.pieces)):
            if self.pieces[i].loss < min_piece:
                min_piece = self.pieces[i].loss
                min_piece_i_1 = min_piece_i_2
                min_piece_i_2 = i
        if np.random.random()<zeta:
            x = self.pieces[min_piece_i_1].piece[np.random.randint(0,5)]
            if np.random.random()>nu:
                y = self.pieces[min_piece_i_2].piece[np.random.randint(0,5)]
            else:
                y = np.random.randint(0,5,3)
        else:
            x = np.random.randint(0,5,3)
            y = np.random.randint(0,5,3)
        self.flip(x,y)
        l = self.loss()
        gamma = np.random.random()
        if l > l_current:
            return
        else:
            if gamma>epsilon:
                self.flip(y,x)
        

    def flip(self, x,y):
        """Flip two pieces.
        
        args:
            x: np.array(3,)
            y: np.array(3,)
        (x and y are the coordinates of the pieces to flip in the cube_cible)"""
        i_px, j_px = indice(self.cube_cible[x[0],x[1],x[2]])
        i_py, j_py = indice(self.cube_cible[y[0],y[1],y[2]])
        self.cube_cible[x[0],x[1],x[2]], self.cube_cible[y[0],y[1],y[2]] = self.cube_cible[y[0],y[1],y[2]], self.cube_cible[x[0],x[1],x[2]]
        self.pieces[i_px].piece[j_px] = y
        self.pieces[i_py].piece[j_py] = x
        self.pieces[i_px].update_loss()
        self.pieces[i_py].update_loss()
    
    def show(self):
        cube_show = np.zeros((5,5,5))
        for i in range(5):
            for j in range(5):
                for k in range(5):
                    cube_show[i,j,k] = self.cube_cible[i,j,k]//5
        print(cube_show)

model_1 = Modelisation()
print("start: ", model_1.cube_cible)
print("start loss: ", model_1.loss())

n = 100000
loss = []
print("start...")
for epoch in range(n):
    curr_loss = model_1.loss()
    if epoch % 10000 == 0:
        print(epoch)
        print("..........")
        print("loss: ", curr_loss)
    if curr_loss > 1.0604:
        break
    loss.append(curr_loss)
    # zeta = 0.8*np.tanh(np.log(epoch+2))
    model_1.step(epsilon = 0.000001, zeta = 0.1, nu = 0.01)

print("end: ", model_1.show())
for i in range(25):
    print(model_1.pieces[i].piece)
    print("end loss: ", model_1.pieces[i].loss)

plt.plot([i for i in range(len(loss))], [1.0605 for i in range(len(loss))], label = "target_loss")
plt.plot(loss)
plt.show()


class QNetwork(torch.nn.Module):
    """
    A Q-Network implemented with PyTorch.

    Attributes
    ----------
    layer1 : torch.nn.Linear
        First fully connected layer.
    layer2 : torch.nn.Linear
        Second fully connected layer.
    layer3 : torch.nn.Linear
        Third fully connected layer.

    Methods
    -------
    forward(x: torch.Tensor) -> torch.Tensor
        Define the forward pass of the QNetwork.
    """

    def __init__(self, n_observations: int, n_actions: int, nn_l1: int, nn_l2: int):
        """
        Initialize a new instance of QNetwork.

        Parameters
        ----------
        n_observations : int
            The size of the observation space.
        n_actions : int
            The size of the action space.
        nn_l1 : int
            The number of neurons on the first layer.
        nn_l2 : int
            The number of neurons on the second layer.
        """
        super(QNetwork, self).__init__()

        self.layer1 = torch.nn.Linear(n_observations,nn_l1)
        self.layer2 = torch.nn.Linear(nn_l1, nn_l2)
        self.layer3 = torch.nn.Linear(nn_l2,n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Define the forward pass of the QNetwork.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor (state).

        Returns
        -------
        torch.Tensor
            The output tensor (Q-values).
        """

        x = self.layer1(x)
        x = torch.nn.functional.relu(x)
        x = self.layer2(x)
        x = torch.nn.functional.relu(x)
        x = self.layer3(x)

        return x
    
    def save(self, path: str):
        """
        Save the model to a file.

        Parameters
        ----------
        path : str
            The path to the file.
        """
        torch.save(self.state_dict(), path)

def test_q_network_agent(env : Modelisation, q_network: torch.nn.Module, num_episode: int = 1, render: bool = True) -> List[int]:
    """
    Test a naive agent in the given environment using the provided Q-network.

    Parameters
    ----------
    env : gym.Env
        The environment in which to test the agent.
    q_network : torch.nn.Module
        The Q-network to use for decision making.
    num_episode : int, optional
        The number of episodes to run, by default 1.
    render : bool, optional
        Whether to render the environment, by default True.

    Returns
    -------
    List[int]
        A list of rewards per episode.
    """
    episode_reward_list = []

    for episode_id in range(num_episode):

        state = env.state()
        done = False
        episode_reward = 0

        while not done:

            # Convert the state to a PyTorch tensor and add a batch dimension (unsqueeze)
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

            action = q_network.forward(state_tensor)
            obs, reward, term, trunc = env.flip(action)
            episode_reward += reward
            done = term or trunc


        episode_reward_list.append(episode_reward)
        print(f"Episode reward: {episode_reward}")

    return episode_reward_list


class EpsilonGreedy:
    """
    An Epsilon-Greedy policy.

    Attributes
    ----------
    epsilon : float
        The initial probability of choosing a random action.
    epsilon_min : float
        The minimum probability of choosing a random action.
    epsilon_decay : float
        The decay rate for the epsilon value after each action.
    env : gym.Env
        The environment in which the agent is acting.
    q_network : torch.nn.Module
        The Q-Network used to estimate action values.

    Methods
    -------
    __call__(state: np.ndarray) -> np.int64
        Select an action for the given state using the epsilon-greedy policy.
    decay_epsilon()
        Decay the epsilon value after each action.
    """

    def __init__(self,
                 epsilon_start: float,
                 epsilon_min: float,
                 epsilon_decay:float,
                 env: gym.Env,
                 q_network: torch.nn.Module):
        """
        Initialize a new instance of EpsilonGreedy.

        Parameters
        ----------
        epsilon_start : float
            The initial probability of choosing a random action.
        epsilon_min : float
            The minimum probability of choosing a random action.
        epsilon_decay : float
            The decay rate for the epsilon value after each episode.
        env : gym.Env
            The environment in which the agent is acting.
        q_network : torch.nn.Module
            The Q-Network used to estimate action values.
        """
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.env = env
        self.q_network = q_network

    def __call__(self, state: np.ndarray) -> np.int64:
        """
        Select an action for the given state using the epsilon-greedy policy.

        If a randomly chosen number is less than epsilon, a random action is chosen.
        Otherwise, the action with the highest estimated action value is chosen.

        Parameters
        ----------
        state : np.ndarray
            The current state of the environment.

        Returns
        -------
        np.int64
            The chosen action.
        """
        action = 0
        t = np.random.random()
        if t < self.epsilon:
          action = np.random.choice(self.env.action_space)
        else:
          action = self.q_network(state)

        return action

    def decay_epsilon(self):
        """
        Decay the epsilon value after each episode.

        The new epsilon value is the maximum of `epsilon_min` and the product of the current
        epsilon value and `epsilon_decay`.
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

class MinimumExponentialLR(torch.optim.lr_scheduler.ExponentialLR):
    def __init__(self, optimizer: torch.optim.Optimizer, lr_decay: float, last_epoch: int = -1, min_lr: float = 1e-6):
        """
        Initialize a new instance of MinimumExponentialLR.

        Parameters
        ----------
        optimizer : torch.optim.Optimizer
            The optimizer whose learning rate should be scheduled.
        lr_decay : float
            The multiplicative factor of learning rate decay.
        last_epoch : int, optional
            The index of the last epoch. Default is -1.
        min_lr : float, optional
            The minimum learning rate. Default is 1e-6.
        """
        self.min_lr = min_lr
        super().__init__(optimizer, lr_decay, last_epoch=-1)

    def get_lr(self) -> list[float]:
        """
        Compute learning rate using chainable form of the scheduler.

        Returns
        -------
        List[float]
            The learning rates of each parameter group.
        """
        return [
            max(base_lr * self.gamma ** self.last_epoch, self.min_lr)
            for base_lr in self.base_lrs
        ]

def train_naive_agent(env: Modelisation,
                      q_network: torch.nn.Module,
                      loss_fn,
                      epsilon_greedy: EpsilonGreedy,
                      device: torch.device,
                      lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
                      num_episodes: int) -> list[float]:
    """
    Train the Q-network on the given environment.

    Parameters
    ----------
    env : gym.Env
        The environment to train on.
    q_network : torch.nn.Module
        The Q-network to train.
    optimizer : torch.optim.Optimizer
        The optimizer to use for training.
    loss_fn : callable
        The loss function to use for training.
    epsilon_greedy : EpsilonGreedy
        The epsilon-greedy policy to use for action selection.
    device : torch.device
        The device to use for PyTorch computations.
    lr_scheduler : torch.optim.lr_scheduler._LRScheduler
        The learning rate scheduler to adjust the learning rate during training.
    num_episodes : int
        The number of episodes to train for.
    gamma : float
        The discount factor for future rewards.

    Returns
    -------
    List[float]
        A list of cumulated rewards per episode.
    """
    episode_reward_list = []

    for episode_index in tqdm(range(1, num_episodes)):
        state, info = env.reset()
        episode_reward = 0

        for t in itertools.count():
            action = epsilon_greedy(state)
            state, reward, term, trunc, info = env.step(action)
            episode_reward += reward
            optimizer.zero_grad()
            optimizer.step()
            lr_scheduler.get_lr()


        episode_reward_list.append(episode_reward)
        epsilon_greedy.decay_epsilon()

    return episode_reward_list