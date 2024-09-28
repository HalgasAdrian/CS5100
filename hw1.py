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

setup(GUI = True, render_delay_sec = 0.000001, gs = 10)


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

# Define the object function to minimize the number of adjacent same-colored cells.
# We will penalize more severe violations. These are areas with large areas of adjacent cells with the same color.
# We will also be rewarding the placement of larger shapes since larger shapes fill up the board more while only costing one shape.

def objective_function(grid, placedShapes):
    violations = 0
    n = len(grid)
    for i in range(n):
        for j in range(n):
            #check horizontal
            if i < n - 1 and grid[i][j] == grid[i + 1][j]:
                violations += 1
            #check vertical
            if j < n - 1 and grid[i][j] == grid[i][j + 1]:
                violations += 1

    # we will add a penalty for the number of shapes placed here
    return violations #+ len(placedShapes)

# Perturb function to explore any neighboring solutions (move brush and place shape)
# This function should help avoid placing shapes on already colored areas by checking if a cell is colored before placing a shape.
# We will also try to target areas with violations by identifying regions of the grid with problems and focusing on fixing those. 

def perturb(grid, current_score):
    actions = ['up', 'down', 'left', 'right', 'switchshape', 'switchcolor', 'place']
    weights = [1, 1, 1, 1, 2, 2, 3]  # Emphasize shape and color switching, and placing
    
    attempt = 0
    max_attempts = 10  # Limit the number of attempts to avoid infinite loops

    while attempt < max_attempts:
        action = random.choices(actions, weights=weights)[0]
        execute(action)
        
        if action in ['up', 'down', 'left', 'right', 'switchshape', 'switchcolor']:
            execute('place')
        
        _, _, _, new_grid, _, _ = execute('export')
        new_score = objective_function(new_grid, placedShapes)
        
        if new_score < current_score:
            print(f"Improvement found with action {action}. New Score: {new_score}")
            break
        attempt += 1

    return new_grid

#def perturb(grid):
    # Randomly change shape, color, or move direction
    violatingCells = []
    n = len(grid)

    # Step 1: Identify all cells with violations (adjacent same-colored cell)
    for i in range(n):
        for j in range(n):
            if i < n - 1 and grid[i][j] == grid[i + 1][j]:
                violatingCells.append((i, j))
            if j < n - 1 and grid[i][j] == grid[i][j+1]:
                violatingCells.append((i, j))
    print(f"Violating cells: {violatingCells}")


    #Step 2: Check for violations, target a violating cell for correction
    if violatingCells:
        # Choose a random violating cell and move the brush there
        cell = random.choice(violatingCells)
        print(f"Moving brush to cell: {cell}")
        moveBrushTo(cell)

        # Change the shape/color and check if the target cell is empty
        current_pos, _, _, grid, _, _ = execute('export')
        if grid[cell[1]][cell[0]] == -1: #-1 represents and empty cell
            print(f"Cell {cell} is empty, placing shape.")
            execute('switchcolor')
            execute('switchshape')
            execute('place')
        else:
            print(f"Cell {cell} is already colored, skipping placement.")
    # Step 3 if no violations, make a random perturbation
    else:
        # Randomly switch shape or color and move the brush
        execute(random.choice(['switchshape', 'switchcolor']))
        execute(random.choice(['up', 'down', 'left', 'right']))
        #execute('place')

        # Check if the new position is empty
        current_pos, _, _, grid, _, _ + execute('export')
        if grid[current_pos[1]][current_pos[0]] == -1:
            execute('place')
        else:
            print(f"Position {current_pos} is already filled, skipping placement.")

    # Step 4: Exporting new grid state
    _, _, _, new_grid, _, _ = execute('export')
    return new_grid

# moveBrushTo helps us move the brush to target areas which need attention for either filling in empty cells or fixing 
# areas which violate our rules.
#

def moveBrushTo(cell):
    current_pos, _, _, _, _, _ = execute('export') # current brush position
    print(f"Current brush position: {current_pos}")

    # Experimental move brush fixes
    maxMoves = 25 #100
    moveCount = 0

    while current_pos != cell:

        # Here we move vertically first 
        if current_pos[1] < cell[1]:
            execute('down')
        elif current_pos[1] > cell[1]:
            execute('up')
        # Then we move horizontally
        if current_pos[0] < cell[0]:
            execute('right')
        elif current_pos[0] > cell[0]:
            execute('left')

        # Fetch the new position after the move
        new_pos, _, _, _, _, _ = execute('export') # update the current brush position
        if new_pos == current_pos:
            print(f"Warning, brush did not update from {current_pos}")
            break #if no position change, break to avoid infinite loop

        current_pos = new_pos
        moveCount += 1 
        print(f"Updated brush position: {current_pos}")
    if moveCount >= maxMoves:
        print("Warning: Exceeded max move attempts")

# Simulated annealing
#
#

def simulated_annealing(initial_grid, max_iterations, initial_temp, cooling_rate, dynamic_reheat_factor):
    current_grid = initial_grid
    _, _, _, _, placedShapes, _ = execute('export')
    current_score = objective_function(current_grid, placedShapes)
    temperature = initial_temp
    last_improvement = 0

    for i in range(max_iterations):
        # Dynamic reheating
        if i - last_improvement > 250:  # Reheat if no improvement for more than 250 iterations
            temperature *= dynamic_reheat_factor
            last_improvement = i
            print(f"Reheating to {temperature} at iteration {i} due to stagnation.")

        new_grid = perturb(current_grid, current_score)
        new_score = objective_function(new_grid, placedShapes)

        # Accept new solution based on simulated annealing criteria
        if new_score < current_score or random.uniform(0, 1) < math.exp((current_score - new_score) / temperature):
            current_grid = new_grid
            current_score = new_score
            last_improvement = i
            print(f"Accepted new state at iteration {i}. Current score: {current_score}")
        else:
            print(f"Rejected change at iteration {i}. Current score: {current_score}, New score: {new_score}")

        # Cool down the temperature
        temperature *= cooling_rate

        if current_score == 0:
            print("Solution found with no violations.")
            break

    return current_grid


#def simulated_annealing(initial_grid, max_iterations, initial_temp, cooling_rate):
#    current_grid = initial_grid
#    _, _, _, _, placedShapes, _ = execute('export')
#    current_score = objective_function(current_grid, placedShapes)
#    temperature = initial_temp
#    print(f"Starting simulated annealing with initial temp {initial_temp}")

#    for i in range(max_iterations):
#        # Perturb the current grid
#        new_grid = perturb(current_grid, current_score)
#        _, _, _, _, placedShapes, _ = execute('export')
#        new_score = objective_function(new_grid, placedShapes)
#        print(f"Current score: {current_score}, New score: {new_score}")
#
 #       # Accept new solution if it is better, or with probability if worse
  #      if new_score < current_score or random.uniform(0, 1) < math.exp((current_score - new_score) / temperature):
   #         current_grid = new_grid
    #        current_score = new_score
     #       print(f"Accepted new state. Current score: {current_score}")
#
 #       # Reduce temperature
  #      temperature *= cooling_rate
#
 #       # Reheat occasionally in case we get stuck, helping us escape local minima
  #      if i % 100 == 0 and temperature < initial_temp:
   #         temperature *= 1.65
    #        print(f"Temp reheated to {temperature}")
#
 #       # Early exit if no violations left
  #      if current_score == 0:
   #         print("Solution found with no violations!")
    #        break
#
 #   return current_grid

# Main function for running our agent
def run_agent():
    initial_temp = 6500
    cooling_rate = 0.97
    max_iterations = 250
    dynamic_reheat_factor = 1.2
    
    # Getting initial state of grid
    _, _, _, initial_grid, _, _ = execute('export')

    print("Running simulated annealing...")

    # Run simulated annealing to optimize the grid
    final_grid = simulated_annealing(initial_grid, max_iterations, initial_temp, cooling_rate, dynamic_reheat_factor)

    # Export final grid to check if constraints are met
    _, _, _, grid, _, done = execute('export')

    if done:
        print("Successfully colored the grid!")
    else:
        print("Failed to meet coloring constraints.")

# Timing the execution 
start = time.time()

# Run the agent
run_agent()

# Save the final state 
end = time.time()

# Save results to files for submission
np.savetxt('grid.txt', grid, fmt = "%d")
with open("shapes.txt", "w") as outfile:
    outfile.write(str(placedShapes))
with open("time.txt", "w") as outfile:
    outfile.write(str(end-start))

########################################

# Do not modify any of the code below. 

########################################

end=time.time()

np.savetxt('grid.txt', grid, fmt="%d")
with open("shapes.txt", "w") as outfile:
    outfile.write(str(placedShapes))
with open("time.txt", "w") as outfile:
    outfile.write(str(end-start))
