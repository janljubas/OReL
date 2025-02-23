import numpy as np

class BankRobberyEnv:
    def __init__(self, grid_size=5, gamma=0.9):
        """
        Grid world with:
        - Banks at fixed positions
        - Police station at (0,0)
        - Agent starts at Bank 1 (grid_size-1, 0)
        """
        self.grid_size = grid_size
        self.banks = [(grid_size-1, 0),  # Bank 1
                      (0, grid_size-1),  # Bank 2
                      (grid_size//2, grid_size//2)]  # Bank 3
        self.police_station = (0, 0)
        
        # State: (agent_pos, police_pos)
        self.nA = 5  # Actions: up, down, left, right, stay
        self.gamma = gamma
        self.reset()

    def reset(self):
        """Initialize positions: agent at Bank 1, police at station"""
        self.agent_pos = (self.grid_size-1, 0)
        self.police_pos = self.police_station
        return (self.agent_pos, self.police_pos)

    def _move(self, pos, action, is_agent=True):
        """Move entity with wall reflection"""
        x, y = pos
        dx, dy = 0, 0
        
        # Agent movement
        if is_agent:
            if action == 0: dx = -1  # up
            elif action == 1: dx = 1  # down
            elif action == 2: dy = -1  # left
            elif action == 3: dy = 1  # right
            # action 4: stay
        
        # Police movement (handled separately)
        new_x = np.clip(x + dx, 0, self.grid_size-1)
        new_y = np.clip(y + dy, 0, self.grid_size-1)
        return (new_x, new_y)

    def _police_movement(self):
        """Stochastic police movement rules"""
        ax, ay = self.agent_pos
        px, py = self.police_pos
        
        # Same row
        if px == ax:
            moves = [(-1,0), (1,0), (0,-1)]  # up, down, left
        # Same column
        elif py == ay:
            moves = [(0,-1), (0,1), (-1,0)]  # left, right, up
        else:
            # Random direction
            moves = [(-1,0), (1,0), (0,-1), (0,1)]
            
        # Choose random move
        dx, dy = moves[np.random.choice(len(moves))]
        return self._move(self.police_pos, (dx, dy), is_agent=False)

    def step(self, action):
        # Agent moves
        new_agent_pos = self._move(self.agent_pos, action)
        
        # Police moves
        new_police_pos = self._police_movement()
        
        # Check collision
        if new_agent_pos == new_police_pos:
            reward = -10000
            done = True
            new_agent_pos = self.banks[0]  # Reset positions
            new_police_pos = self.police_station
        else:
            reward = 100000 if (new_agent_pos in self.banks 
                               and new_agent_pos != new_police_pos) else 0
            done = False
            
        self.agent_pos = new_agent_pos
        self.police_pos = new_police_pos
        return (new_agent_pos, new_police_pos), reward, done, {}
    

if __name__ == "__main__":
    print("We started!")
    
    env = BankRobberyEnv(grid_size=5, gamma=0.9)
    state = env.reset()  # Agent at (4,0), Police at (0,0)
    print("Initial state:", state)

    # Agent chooses to move left (action=2)
    next_state, reward, done, _ = env.step(2)
    print("Next state:", next_state, "| Reward:", reward, "| Done:", done)
