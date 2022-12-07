import pygame
import enum
import sys
import torch
from copy import deepcopy
from random import choice, randrange, random
from QNN import QLearningNeuralNetwork
'''
Authors: Karen Zamora, Josef Ruzicka
         B37741      , B87095
         
Useful References (Some ideas where adapted from a couple of these):
    https://openreview.net/pdf?id=8TLyqLGQ7Tg
    https://medium.com/analytics-vidhya/introduction-to-reinforcement-learning-q-learning-by-maze-solving-example-c34039019317
    https://www.youtube.com/watch?v=PJl4iabBEz0
    https://www.youtube.com/watch?v=z4OomBu6kD0
'''


# Tetris related
W, H = 10, 20
TILE = 35
GAME_RES = W * TILE, H * TILE
RES = 750, 940
FPS = 60

# Rewards
RECORD_REWARD = 100
LINE_REWARD = 50
LOWER_ROWS_REWARD = 10
MIDDLE_ROWS_REWARD = 0
UPPER_ROWS_REWARD = -10

# Learning parameters
DEFAULT_ALPHA_LR = 0.03 # learning rate
DEFAULT_GAMMA_DISCOUNT = 0.3 # discount
DEFAULT_EPS_GREEDY  = 1
DEFAULT_ETA_DECAY  = 0.001

RANDOM_SEED = 2
EPOCHS = 3

class Action(enum.IntEnum):
    LEFT = 0
    RIGHT = 1    
    ROTATE = 2
    #DOWN = 3

class Agent():
    # Initializes the agent
    def __init__(self,seed,state_dims,actions,learning_rate,discount_factor,eps_greedy,decay):
        # Use self.prng any time you require to call a random funcion
        self.prng = random.Random()
        self.prng.seed(seed)
        #self.state_dims = state_dims
        self.actions = actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.eps_greedy = eps_greedy
        self.decay = decay
        
        '''
        Explicación: nuestro programa se basa en recompenzar la colocación de piezas
        a la menor altura posible, entonces nuestros estados consisten en un arreglo 
        con las 10 alturas máximas de cada columna, pero hacer todas las combinaciones
        posibles de alturas de cada columnas requeriría demasiada memoria: 
            10 columnas, y valores entre 0 y 20 para cada una representando su altura
        por lo tanto ignoraremos el eje x, y nuestra tabla solo se tratará de dónde estamos
        en el eje y, recompenzando solo basado en altura (también habrá recompenzas por
        eliminación de filas y castigos por perder el juego).
        '''
        #qtable as tensor [height0[L,R,Rotate], height1[L,R,Rotate], ..., height19[L,R,Rotate]]
        self.qtable = []
        for col in range(H):
            current_height = []
            for a in actions:
                # note: 0:Left, 1:Right, 2:Rotate
                current_height.append(0)
            self.qtable.append(current_height)
        #print(self.qtable)
        pass
    
    # Performs a complete simulation by the agent
    def simulation(self, env):
        env.reset()
        while (not env.is_terminal_state()):
            self.step(env, learn=True)
        self.eps_greedy -= self.eps_greedy * self.decay 
        pass
    
    # Performs a single step of the simulation by the agent, if learn=False no updates are performed
    def step(self, env, learn=True):
        #print(self.eps_greedy)
        #Exploration mode
        r_mode = self.prng.random()
        #print('r', r_mode)
        # TODO: sometimes this if wont work properly
        if (learn and (r_mode < self.eps_greedy)):
            action_value = self.prng.random() * 3
            #print('exploring')
            if ( action_value < 1):
                a = 0
            elif ( action_value < 2):
                a = 1
            elif ( action_value < 3):
                a = 2
            #print('a', a)
            
            # Q function = Q(s,a) = R(T(s,a)) + γ · maxa’ Q(T(s,a),a’)
            # Q(s,a) ← Q(s,a) + α(R(T(s,a)) + γ maxa’ Q(T(s,a),a’) − Q(s,a))
            
            '''
                Get max col height: 
                (get state, take first 10 values which correspond to col heights, select highest value and use it as qtable index)
            '''
            state = env.get_current_state()
            max_height = max(state[:10])
            
            QSA  = self.qtable[max_height][a]
            #RTSA = env.perform_action(a)[a]
            #maxa = self.qtable[env.position[0]][env.position[1]].index(max(self.qtable[env.position[0]][env.position[1]]))
            #QRSA = env.perform_action(maxa)[0]
                
            #QSA = QSA + DEFAULT_ALPHA_LR *   (RTSA + DEFAULT_GAMMA_DISCOUNT * QTSA - QSA)
            self.qtable[max_height][a] +=\
            DEFAULT_ALPHA_LR *(\
            env.perform_action(a)[0] +\
            DEFAULT_GAMMA_DISCOUNT *\
            max(self.qtable[max_height]) -\
            QSA)
            
        # exploitation mode
        else:
            #print('exploitation')
            best_known_action = self.qtable[max_height].index(max(self.qtable[max_height]))
            env.perform_action(best_known_action)
        #print(self.qtable)
        pass

'''
# Attempt at solving the problem using Deep Q Learning.
# There was some confusion towards connecting the NN to the agent and the game.
class Deep_Q_Learning_Agent():
    def __init__(self,memory_capacity, batch_size, iters, learning_rate, discount, eps_greedy, decay):
        #random.seed(RANDOM_SEED)
        self.prng = random()
        #self.prng.seed(RANDOM_SEED)
        self.max_memory = memory_capacity
        self.memory = []
        # Batch size for each training
        self.batch_size = batch_size
        # Number of training sessions before weigths NN update 
        self.iters = iters
        self.learning = learning_rate
        self.discount = discount # gamma
        self.eps = eps_greedy
        self.decay = decay
        #self.model = QLearningNeuralNetwork()
        #self.my_trainer = QTrainer(self.model,self.learning,self.discount)

    def train(self):#?
    '''
'''
        self.my_trainer.train(state,action,reward,next_state)
        self.memory.append((state, action, reward, next_state))'''
        #pass
'''
    def simulation(self,tetris):
        tetris.reset()
        epoch = 0
        while epoch < EPOCHS:
            while not tetris.is_game_over():
                self.step(tetris)
            epoch += 1
        

    def step(self,env,learn=True):
        reward = 0
        action = None
        #rand = self.prng.random()  
        rand = random()
        rand = 0.20
        learn = True
        if rand > self.eps or learn==False:
            # Best known action
            old_state = env.get_state()
            state = torch.FloatTensor(old_state)
            pred = self.model(state)
            '''
'''
            state = torch.tensor(old_state, dtype=torch.float)
            
            pred = self.model(state)
            action = torch.argmax(pred).item()            
            reward,new_state = env.perform_action(action)            
            self.train_memory(old_state, action, reward, new_state)'''            
'''
        elif learn and rand < self.eps:
            # Random action
            action = int(random()*3)
            #reward,new_state = env.perform_action(action)
            env.perform_action(action)
            '''
class Tetris:

    grid = [pygame.Rect(x * TILE, y * TILE, TILE, TILE) for x in range(W) for y in range(H)]

    figures_pos = [[(-1, 0), (-2, 0), (0, 0), (1, 0)],
                [(0, -1), (-1, -1), (-1, 0), (0, 0)],
                [(-1, 0), (-1, 1), (0, 0), (0, -1)],
                [(0, 0), (-1, 0), (0, 1), (-1, -1)],
                [(0, 0), (0, -1), (0, 1), (-1, -1)],
                [(0, 0), (0, -1), (0, 1), (1, -1)],
                [(0, 0), (0, -1), (0, 1), (-1, 0)]]
    figures = [[pygame.Rect(x + W // 2, y + 1, 1, 1) for x, y in fig_pos] for fig_pos in figures_pos]
    figure_rect = pygame.Rect(0, 0, TILE - 2, TILE - 2)
    anim_count, anim_speed, anim_limit = 0, 60, 2000
    

    get_color = lambda : (randrange(30, 256), randrange(30, 256), randrange(30, 256))

    figure, next_figure = deepcopy(choice(figures)), deepcopy(choice(figures))
    color, next_color = get_color(), get_color()

    score, lines = 0, 0
    scores = {0: 0, 1: 100, 2: 300, 3: 700, 4: 1500}

    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode(RES)
        self.game_screen = pygame.Surface(GAME_RES)
        self.clock = pygame.time.Clock()
        self.board = [[0 for i in range(W)] for j in range(H)]
        self.score, self.lines = 0, 0    
        self.record =0
        self.main_font = pygame.font.Font('fonts/arcade.ttf', 65)
        self.font = pygame.font.Font('fonts/mario.ttf', 45)
        self.title_tetris = self.main_font.render('TETRIS', True, pygame.Color('darkorange'))
        self.title_score = self.font.render('score:', True, pygame.Color('green'))
        self.title_record = self.font.render('record:', True, pygame.Color('purple'))
        self.figure_land  = False

    def get_color(self):
        return lambda : (randrange(30, 256), randrange(30, 256), randrange(30, 256))
    
    def check_borders(self,figure):
        for i in range(4):
            if figure[i].x < 0 or figure[i].x > W - 1:
                return False
            elif figure[i].y > H - 1 or self.board[figure[i].y][figure[i].x]:
                return False
        return True

    def get_record(self):
        try:
            with open('record') as f:
                return f.readline()
        except FileNotFoundError:
            with open('record', 'w') as f:
                f.write('0')

    def set_record(self,record, score):
        rec = max(int(record), score)
        print("rec = ",int(record)," score= ",score)
        print("rec= ",rec)
        with open('record', 'w') as f:
            f.write(str(rec))

    def move_horizontally(self,dx):
        figure_old = deepcopy(self.figure)
        for i in range(4):
            self.figure[i].x += dx
            if not self.check_borders(self.figure):
                self.figure = deepcopy(figure_old)
                break

    def move_vertically(self):
        self.figure_land = False
        self.anim_count += self.anim_speed
        if self.anim_count > self.anim_limit:
            self.anim_count = 0
            figure_old = deepcopy(self.figure)
            for i in range(4):
                self.figure[i].y += 1
                if not self.check_borders(self.figure):
                    for i in range(4):
                        self.board[figure_old[i].y][figure_old[i].x] = self.color
                    self.figure, self.color = self.next_figure, self.next_color
                    self.figure_land = True
                    self.next_figure, self.next_color = deepcopy(choice(self.figures)), self.get_color()
                    self.anim_limit = 2000
                    break

    def rotate(self,rotate):
        center = self.figure[0]
        figure_old = deepcopy(self.figure)
        if rotate:
            for i in range(4):
                x = self.figure[i].y - center.y
                y = self.figure[i].x - center.x
                self.figure[i].x = center.x - x
                self.figure[i].y = center.y + y
                if not self.check_borders(self.figure):
                    self.figure = deepcopy(figure_old)
                    break

    def check_lines(self):
        line, lines = H - 1, 0
        for row in range(H - 1, -1, -1):
            count = 0
            for i in range(W):
                if self.board[row][i]:
                    count += 1
                self.board[line][i] = self.board[row][i]
            if count < W:
                line -= 1
            else:
                self.anim_speed += 3
                lines += 1
        return lines

    def calculate_figure_position_reward(self, state):
        highest_row = 0
        for i in range(9):
            if state[i]>highest_row:
                highest_row= state[i]
        return highest_row

    def get_current_state(self):
        # get board state. returns the height of each column.
        state = [-1]*W
        for lineIndex, line in enumerate(self.board):
            for cellIndex, cell in enumerate(line):
                if state[cellIndex] == -1 and cell != 0:
                    state[cellIndex] = H - lineIndex
        # get figure state
        for coordenate in self.figure:
            state.append(coordenate[0])
            state.append(coordenate[1])

        return state

    def is_game_over(self):
        game_over = False
        for i in range(W):
            if self.board[0][i]:
                self.set_record(self.record, self.score)
                self.reset()
                game_over= True
                for i_rect in self.grid:
                    # TODO: stop this from redrawing
                    pygame.draw.rect(self.game_screen, (181,59,183), i_rect)
                    self.screen.blit(self.game_screen, (20, 20))
                    pygame.display.flip()
                    self.clock.tick(200)
        return game_over
                    
    def reset(self):
        self.board = [[0 for i in range(W)] for i in range(H)]
        self.anim_count, self.anim_speed, self.anim_limit = 0, 60, 2000
        self.score = 0

    def perform_action(self,action):
        reward = 0
        #while True:
        record = self.get_record()
        dx, rotate = 0, False
        self.screen.fill((0,0,0))
        self.screen.blit(self.game_screen, (20, 20))
        self.game_screen.fill((0,0,0))
        # delay for full lines
        for i in range(self.lines):
            pygame.time.wait(200)
        # controls
        #self.controls
        for event in pygame.event.get():
            if event.type == pygame.QUIT:   
                pygame.display.quit()
                pygame.quit()
                sys.exit()
            '''if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    dx = -1
                elif event.key == pygame.K_RIGHT:
                    dx = 1
                elif event.key == pygame.K_DOWN:
                    self.anim_limit = 100
                elif event.key == pygame.K_UP:
                    rotate = True'''

        if action == 0:
            dx=-1
        elif action == 1:
            dx = 1
        elif action ==2:
            rotate = True
        elif action == 3:
            self.anim_limit = 100
        #self.move_horizontally(dx)
        self.move_horizontally(dx)

        # Vertically
        self.move_vertically()
        # rotate
        self.rotate(rotate)
        lines = self.check_lines()
        if lines > 0:
            reward += LINE_REWARD        
        # draw grid
        [pygame.draw.rect(self.game_screen, (40, 40, 40), i_rect, 1) for i_rect in self.grid]
        # draw figure
        for i in range(4):
            self.figure_rect.x = self.figure[i].x * TILE
            self.figure_rect.y = self.figure[i].y * TILE
            pygame.draw.rect(self.game_screen, (181,59,183), self.figure_rect)
        # draw field
        for y, raw in enumerate(self.board):
            for x, col in enumerate(raw):
                if col:
                    self.figure_rect.x, self.figure_rect.y = x * TILE, y * TILE
                    pygame.draw.rect(self.game_screen, (181,59,183), self.figure_rect)
        # draw next figure
        for i in range(4):
            self.figure_rect.x = self.next_figure[i].x * TILE + 380
            self.figure_rect.y = self.next_figure[i].y * TILE + 185
            pygame.draw.rect(self.screen, (181,59,183), self.figure_rect)
        # draw titles
        self.screen.blit(self.title_tetris, (485, -10))
        self.screen.blit(self.title_score, (535, 780))
        self.screen.blit(self.font.render(str(self.score), True, pygame.Color('white')), (550, 840))
        self.screen.blit(self.title_record, (525, 650))
        self.screen.blit(self.font.render(record, True, pygame.Color('gold')), (550, 710))
        # game over
        #self.is_game_over()
        pygame.display.flip()
        self.clock.tick(FPS)
        next_state = self.get_current_state()
        if self.figure_land:
            highest_row = self.calculate_figure_position_reward(next_state)            
            if highest_row <6:
                reward+= LOWER_ROWS_REWARD
            elif highest_row >=6 and highest_row<11:
                reward+= MIDDLE_ROWS_REWARD
            elif highest_row >10:
                
                reward+= UPPER_ROWS_REWARD
            self.score+= reward
            '''
            if self.score> int(record):
                reward+= RECORD_REWARD
                self.score += RECORD_REWARD'''
        return reward, next_state

class Game:
    def __init__(self):
        self.memory_capacity = 10000
        self.batch_size = 1000
        self.iters = 10
        self.lr = 0.001
        self.discount = 0.25
        self.greedy = 0.25
        self.decay = 1e-7
        self.tetris = Tetris()
        self.agent = Agent(self.memory_capacity,self.batch_size,self.iters,self.lr,self.discount,self.greedy,self.decay)

    def play(self):
        self.agent.simulation(self.tetris)
        

if __name__ == "__main__":
    game = Game()
    game.play()
    