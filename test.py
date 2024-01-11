import time
from tkinter import Frame, Label, CENTER
import numpy as np
from game import TwntyFrtyEight
from agent import Agent
import constants as c

def main():
    game = TwntyFrtyEight()
    game_grid = GameGrid(game)
    agent = Agent(game)
    agent.w = np.load('w_star.npy')
    
    while True:
        current_state = TwntyFrtyEight.board_to_state(game_grid.board)
        if TwntyFrtyEight.is_terminal_state(current_state):
            break
        action = agent.softmax_policy(current_state)
        
        game_grid.swipe(action)
        game_grid.update_idletasks()
        game_grid.update()
        time.sleep(0.2)
        
    game_grid.mainloop()

class GameGrid(Frame):
    def __init__(self, game: TwntyFrtyEight):
        Frame.__init__(self)
        
        self.game = game

        self.grid()
        self.master.title('2048')
        self.master.bind("<Key>", self.key_down)

        self.commands = {
            0: game.left,
            1: game.up,
            2: game.right,
            3: game.down,
            c.KEY_UP: game.up,
            c.KEY_DOWN: game.down,
            c.KEY_LEFT: game.left,
            c.KEY_RIGHT: game.right,
            c.KEY_UP_ALT1: game.up,
            c.KEY_DOWN_ALT1: game.down,
            c.KEY_LEFT_ALT1: game.left,
            c.KEY_RIGHT_ALT1: game.right,
            c.KEY_UP_ALT2: game.up,
            c.KEY_DOWN_ALT2: game.down,
            c.KEY_LEFT_ALT2: game.left,
            c.KEY_RIGHT_ALT2: game.right,
        }

        self.grid_cells = []
        self.init_grid()
        self.board = game.new_board(c.GRID_LEN, c.GRID_LEN)
        self.board_history = []
        self.update_grid_cells()
        
        self.update_idletasks()
        self.update()
        # self.mainloop()

    def init_grid(self):
        background = Frame(self, bg=c.BACKGROUND_COLOR_GAME,width=c.SIZE, height=c.SIZE)
        background.grid()

        for i in range(c.GRID_LEN):
            grid_row = []
            for j in range(c.GRID_LEN):
                cell = Frame(
                    background,
                    bg=c.BACKGROUND_COLOR_CELL_EMPTY,
                    width=c.SIZE / c.GRID_LEN,
                    height=c.SIZE / c.GRID_LEN
                )
                cell.grid(
                    row=i,
                    column=j,
                    padx=c.GRID_PADDING,
                    pady=c.GRID_PADDING
                )
                t = Label(
                    master=cell,
                    text="",
                    bg=c.BACKGROUND_COLOR_CELL_EMPTY,
                    justify=CENTER,
                    font=c.FONT,
                    width=5,
                    height=2)
                t.grid()
                grid_row.append(t)
            self.grid_cells.append(grid_row)

    def update_grid_cells(self):
        for i in range(c.GRID_LEN):
            for j in range(c.GRID_LEN):
                new_number = self.board[i][j]
                if new_number == 0:
                    self.grid_cells[i][j].configure(text="",bg=c.BACKGROUND_COLOR_CELL_EMPTY)
                else:
                    self.grid_cells[i][j].configure(
                        text=str(new_number),
                        bg=c.BACKGROUND_COLOR_DICT[new_number],
                        fg=c.CELL_COLOR_DICT[new_number]
                    )
        self.update_idletasks()

    def key_down(self, event):
        key = event.keysym
        print(event)
        if key == c.KEY_QUIT: exit()
        if key == c.KEY_BACK and len(self.board_history) > 1:
            self.board = self.board_history.pop()
            self.update_grid_cells()
            print('back on step total step:', len(self.board_history))
        elif key in self.commands:
            self.swipe(key)
    
    def swipe(self, key):
        compressed_board, points = self.commands[key](self.board)
        done = np.any(self.board != compressed_board)
        if done:
            self.board = self.game.add_two(compressed_board)
            # record last move
            self.board_history.append(self.board)
            self.update_grid_cells()
            # if self.game.get_board_status(self.board) == 'win':
            #     self.grid_cells[1][1].configure(text="You", bg=c.BACKGROUND_COLOR_CELL_EMPTY)
            #     self.grid_cells[1][2].configure(text="Win!", bg=c.BACKGROUND_COLOR_CELL_EMPTY)
            # if self.game.get_board_status(self.board) == 'lose':
            #     self.grid_cells[1][1].configure(text="You", bg=c.BACKGROUND_COLOR_CELL_EMPTY)
            #     self.grid_cells[1][2].configure(text="Lose!", bg=c.BACKGROUND_COLOR_CELL_EMPTY)
                    
if __name__ == '__main__':
    main()
