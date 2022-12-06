import pygame
from copy import deepcopy
from random import choice, randrange

W, H = 10, 20
TILE = 45
GAME_RES = W * TILE, H * TILE
RES = 750, 940
FPS = 60

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
        self.main_font = pygame.font.Font('fonts/arcade.ttf', 65)
        self.font = pygame.font.Font('fonts/mario.ttf', 45)
        self.title_tetris = self.main_font.render('TETRIS', True, pygame.Color('darkorange'))
        self.title_score = self.font.render('score:', True, pygame.Color('green'))
        self.title_record = self.font.render('record:', True, pygame.Color('purple'))

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
        with open('record', 'w') as f:
            f.write(str(rec))

    def controls(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    dx = -1
                elif event.key == pygame.K_RIGHT:
                    dx = 1
                elif event.key == pygame.K_DOWN:
                    anim_limit = 100
                elif event.key == pygame.K_UP:
                    rotate = True

    def move_horizontally(self,dx):
        figure_old = deepcopy(self.figure)
        for i in range(4):
            self.figure[i].x += dx
            if not self.check_borders(self.figure):
                self.figure = deepcopy(figure_old)
                break

    def move_vertically(self):
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

    def calculate_score(self,lines):
        self.score += self.scores[lines]

    def is_game_over(self,record):
        for i in range(W):
            if self.board[0][i]:
                self.set_record(record, self.score)
                self.board = [[0 for i in range(W)] for i in range(H)]
                self.anim_count, self.anim_speed, self.anim_limit = 0, 60, 2000
                self.score = 0
                for i_rect in self.grid:
                    pygame.draw.rect(self.game_screen, (181,59,183), i_rect)
                    self.screen.blit(self.game_screen, (20, 20))
                    pygame.display.flip()
                    self.clock.tick(200)

    def run(self):
        while True:
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
                    exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        dx = -1
                    elif event.key == pygame.K_RIGHT:
                        dx = 1
                    elif event.key == pygame.K_DOWN:
                        self.anim_limit = 100
                    elif event.key == pygame.K_UP:
                        rotate = True
            #self.move_horizontally(dx)
            self.move_horizontally(dx)

            # Vertically
            self.move_vertically()
            # rotate
            self.rotate(rotate)
            lines = self.check_lines()
            # compute score
            self.calculate_score(lines)
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
            self.is_game_over(record)
            pygame.display.flip()
            self.clock.tick(FPS)

tetris = Tetris()
tetris.run()
