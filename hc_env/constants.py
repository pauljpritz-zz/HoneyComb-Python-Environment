# Agent numbers
NUM_UNINFORMED = 8
NUM_INFORMED = 2

# Actions
UP = (0, -1)
DOWN = (0, 1)
UP_LEFT = (-1, -1)
UP_RIGHT = (1, 0)
DOWN_LEFT = (-1, 0)
DOWN_RIGHT = (1, 1)
PASS = (0, 0)

ACTIONS = [UP, DOWN, UP_LEFT, UP_RIGHT, DOWN_LEFT, DOWN_RIGHT, PASS]
ACTIONS_NAME = ["UP", "DOWN", "UP_LEFT", "UP_RIGHT", "DOWN_LEFT", "DOWN_RIGHT", "PASS"]

# Matrix size including void
PAYOFF_LOCATIONS = [(-3, -6), (3, -3), (6, 3), (3, 6), (-3, 3), (-6, -3)]
START_POSITION = (0, 0)
