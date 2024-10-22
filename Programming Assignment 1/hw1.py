import time
import numpy as np
import random
import math
from gridgame import *

##############################################################################################################################

# You can visualize what your code is doing by setting the GUI argument in the following line to true.
# The render_delay_sec argument allows you to slow down the animation, to be able to see each step more clearly.

# For your final submission, please set the GUI option to False.

# The gs argument controls the grid size. You should experiment with various sizes to ensure your code generalizes.

##############################################################################################################################

setup(GUI = True, render_delay_sec = 0.00001, gs = 7)


##############################################################################################################################

# Initialization

# shapePos is the current position of the brush.

# currentShapeIndex is the index of the current brush type being placed (order specified in gridgame.py, and assignment instructions).

# currentColorIndex is the index of the current color being placed (order specified in gridgame.py, and assignment instructions).

# grid represents the current state of the board. 
    
    # -1 indicates an empty cell
    # 0 indicates a cell colored in the first color (indigo by default)
    # 1 indicates a cell colored in the second color (taupe by default)
    # 2 indicates a cell colored in the third color (veridian by default)
    # 3 indicates a cell colored in the fourth color (peach by default)

# placedShapes is a list of shapes that have currently been placed on the board.
    
    # Each shape is represented as a list containing three elements: a) the brush type (number between 0-8), 
    # b) the location of the shape (coordinates of top-left cell of the shape) and c) color of the shape (number between 0-3)

    # For instance [0, (0,0), 2] represents a shape spanning a single cell in the color 2=veridian, placed at the top left cell in the grid.

# done is a Boolean that represents whether coloring constraints are satisfied. Updated by the gridgames.py file.

##############################################################################################################################

shapePos, currentShapeIndex, currentColorIndex, grid, placedShapes, done = execute('export')

#input()   # <-- workaround to prevent PyGame window from closing after execute() is called, for when GUI set to True. Uncomment to enable.
print(shapePos, currentShapeIndex, currentColorIndex, grid, placedShapes, done)


####################################################
# Timing your code's execution for the leaderboard.
####################################################

start = time.time()  # <- do not modify this.

##########################################
# Write all your code in the area below. 
##########################################

# CHATGPT was used during this assignment to help me understand how to implement local search, in this case, simulated annealing to this specific problem.
# I do not have much experience using python in AI so it helped me learn the basics and debug, I faced issues with brushes not moving, lots of crashing, and lots of the agent not following rules I put in place.
# So just about every function has been checked with generative AI and some of the design choices were implemented and helped me complete this assignment. 
# - Adrian Halgas 

# Defining the object function to minimize the number of adjacent same-colored cells.
# We will penalize violations. These are areas with adjacent cells with the same color.
# Once we have no violations and no empty spaces we will set our score to float('inf'), this helps the program terminate.

def objective_function(grid, placedShapes):
    base_score = 10000  # Start with a high base score
    penalty_empty = 150  # Penalty for each empty cell
    penalty_adjacent = 1500  # Penalty for adjacent same-colored cells
    reward_placement = 300  # Reward for each non-violating placement

    n = len(grid)
    num_empty = sum(1 for row in grid for cell in row if cell == -1)
    num_violations = 0
    num_placements = len(placedShapes)

    for i in range(n):
        for j in range(n):
            if grid[i][j] != -1:
                if j + 1 < n and grid[i][j + 1] == grid[i][j]:
                    num_violations += 1
                if i + 1 < n and grid[i + 1][j] == grid[i][j]:
                    num_violations += 1

    score = (base_score - penalty_empty * num_empty - penalty_adjacent * num_violations + reward_placement * num_placements)
    
    # If no empty cells and no violations, grid is perfectly solved
    if num_empty == 0 and num_violations == 0:
        return float('inf')  # or any special flag value that indicates completion
    
    return score


# Perturb function to explore any neighboring solutions (move brush and place shape)
# We will also try to target areas with violations by identifying regions of the grid with problems and focusing on fixing those. 
# Perturb will also use helper functions is_valid_placement, adjacent, and get_shape_coverage to help determine if the 
# shape being placed will violate our rules, if so, perturb can undo up to 10 consecutive times to fix the placement. 

def perturb(grid, current_score):
    actions = ['up', 'down', 'left', 'right', 'switchshape', 'switchcolor', 'place']
    weights = [1, 1, 1, 1, 1, 2, 3]  # Emphasize shape and color switching, and placing
    max_attempts = 25
    undo_counter = 0
    max_undos = 10  # Limit the number of consecutive undos

    valid_states = [(grid.copy(), current_score)]  # Store valid states to backtrack if needed

    for _ in range(max_attempts):
        action = random.choices(actions, weights=weights)[0]
        execute(action)

        if action in ['up', 'down', 'left', 'right', 'switchshape', 'switchcolor']:
            execute('place')

        _, _, _, new_grid, new_placedShapes, _ = execute('export')

        if action == 'place':
            if not is_valid_placement(new_grid, new_placedShapes):
                print("Violation detected")
                if undo_counter < max_undos:
                    execute('undo')
                    undo_counter += 1
                    print(f"Violation undone, undones used: {undo_counter}")
                    continue
                else:
                    print("Max undos reached, trying alternative moves")
                    # If max undos reached, backtrack to the last known good state
                    if valid_states:
                        new_grid, _ = valid_states.pop()
                        continue
                    else:
                        break  # No valid states left, fully reset if necessary

        new_score = objective_function(new_grid, new_placedShapes)
        if new_score > current_score:
            print(f"Improvement found or state maintained with action {action}. New Score: {new_score}")
            valid_states.append((new_grid.copy(), new_score))  # Store new valid state
            return new_grid  # Accept the new state if it improves the score

    return new_grid  # Return the original grid if no acceptable move was found

# Helper function for perturb, see above. 

def is_valid_placement(grid, placedShapes):
    n = len(grid)  # Assumes grid is square

    for i in range(n):
        for j in range(n):
            # Check adjacent cells for the same color
            if grid[i][j] != -1:  # Check only non-empty cells
                if j + 1 < n and grid[i][j] == grid[i][j + 1]:
                    return False  # Right cell match
                if i + 1 < n and grid[i][j] == grid[i + 1][j]:
                    return False  # Bottom cell match

    return True  # No violations found

# Helper function for perturb, see above. 

def adjacent(coord1, coord2):
    x1, y1 = coord1
    x2, y2 = coord2
    return abs(x1 - x2) <= 1 and abs(y1 - y2) <= 1 and not (x1 == x2 and y1 == y2)

# Helper function for perturb, see above. 

def get_shape_coverage(shape_array, x, y):
    # Calculate the grid cells covered by a shape based on its top-left position (x, y)
    shape_coverage = []
    for dx in range(shape_array.shape[0]):
        for dy in range(shape_array.shape[1]):
            if shape_array[dx, dy] == 1:  # Check if the cell is part of the shape (marked by '1')
                shape_coverage.append((x + dx, y + dy))
    return shape_coverage

# Shape definitions represented by arrays
shapes = [
    np.array([[1]]),  # Shape 0: 1x1 square
    np.array([[1, 0], [0, 1]]),  # Shape 1: 2x2 square with diagonal holes
    np.array([[0, 1], [1, 0]]),  # Shape 2: 2x2 square with diagonal holes (transpose)
    np.array([[1, 0], [0, 1], [1, 0], [0, 1]]),  # Shape 3: 2x4 rectangle with holes
    np.array([[0, 1], [1, 0], [0, 1], [1, 0]]),  # Shape 4: 2x4 rectangle with holes (transpose)
    np.array([[1, 0, 1, 0], [0, 1, 0, 1]]),  # Shape 5: 4x2 rectangle with alternating holes
    np.array([[0, 1, 0, 1], [1, 0, 1, 0]]),  # Shape 6: 4x2 rectangle with alternating holes (transpose)
    np.array([[0, 1, 0], [1, 0, 1]]),  # Shape 7: Sparse T-shape
    np.array([[1, 0, 1], [0, 1, 0]])  # Shape 8: Sparse T-shape (reversed)
]

# Simulated annealing
# This is our local search implementation, I chose simulated annealing because I hoped that it would help us by starting the agent to be
# aggressive in placing shapes in order to fill up the grid faster, and slowing down to be more precise towards the end. 

def simulated_annealing(initial_grid, max_iterations, initial_temp, cooling_rate, dynamic_reheat_factor):
    current_grid = initial_grid
    _, _, _, _, placedShapes, done = execute('export')
    current_score = objective_function(current_grid, placedShapes)
    temperature = initial_temp
    last_improvement = 0

    for i in range(max_iterations):
        if current_score == float('inf'):
            print("Perfect solution found, terminating early.")
            return current_grid, True

        new_grid = perturb(current_grid, current_score)
        new_score = objective_function(new_grid, placedShapes)

        if new_score == float('inf'):
            print("Perfect solution found during iteration, terminating early.")
            return new_grid, True

        if new_score < current_score or random.uniform(0, 1) < math.exp((current_score - new_score) / temperature):
            current_grid = new_grid
            current_score = new_score
            last_improvement = i
            print(f"Accepted new state at iteration {i}. Current score: {current_score}")
        else:
            print(f"Rejected change at iteration {i}. Current score: {current_score}, New score: {new_score}")

        temperature *= cooling_rate

    return current_grid, False  # Ensure a boolean for 'done' is always returned


# Main function for running our agent
def run_agent():
    initial_temp = 5400
    cooling_rate = 0.999
    max_iterations = 50000
    dynamic_reheat_factor = 1.5
    
    # Getting initial state of grid
    _, _, _, initial_grid, _, _ = execute('export')

    print("Running simulated annealing...")

    # Run simulated annealing to optimize the grid
    final_grid, done = simulated_annealing(initial_grid, max_iterations, initial_temp, cooling_rate, dynamic_reheat_factor)

    # Check if the simulation was terminated because the grid was successfully solved
    if done:
        print("Successfully colored the grid with no violations!")
    else:
        print("Failed to meet coloring constraints or ran out of iterations.")

    # Export final grid to check if constraints are met externally
    _, _, _, grid, _, _ = execute('export')
    # The above line re-fetches 'done', but it's not used afterward, which may cause confusion.

# Run the agent
run_agent()

########################################

# Do not modify any of the code below. 

########################################

end=time.time()

np.savetxt('grid.txt', grid, fmt="%d")
with open("shapes.txt", "w") as outfile:
    outfile.write(str(placedShapes))
with open("time.txt", "w") as outfile:
    outfile.write(str(end-start))
