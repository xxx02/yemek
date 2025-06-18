import numpy as np
from PIL import Image, ImageGrab
import time
import pydirectinput  # For simulating keyboard presses

# --- Tetris Game Constants from main.py ---
TOP_LEFT = (549, 488)  # Top-left pixel coordinate of the Tetris grid
BOTTOM_RIGHT = (1189, 1857)  # Bottom-right pixel coordinate of the Tetris grid
GRID_WIDTH = 10
GRID_HEIGHT = 21  # Common Tetris height, including one row above visible area for spawning
WHITE = (255, 255, 255)  # RGB value for empty cells (white background)

# Coordinates of the "next pieces" preview areas
NEXT_REGIONS = {
    1: ((1244, 689), (1371, 970)),
    2: ((1244, 970), (1371, 1251)),
    3: ((1244, 1251), (1371, 1532))
}

# Average RGB colors for each Tetris piece (for identifying pieces from screenshot)
PIECE_COLORS = {
    'T': (255, 202, 255),  # Purple
    'O': (51, 169, 236),  # Yellow
    'I': (204, 110, 120),  # Cyan
    'J': (254, 114, 254),  # Blue
    'L': (252, 128, 131),  # Orange
    'Z': (50, 180, 121),  # Red
    'S': (255, 164, 189)  # Green
}

SPAWN_POSITIONS = {
        'I': [3, 4, 3, 5],
        'O': [4, 4, 4, 4],
        'T': [3, 4, 3, 3],
        'L': [3, 4, 3, 3],
        'J': [3, 4, 3, 3],
        'S': [3, 4, 3, 3],
        'Z': [3, 4, 3, 3],
    }

# Cell size in pixels (calculated from grid coordinates)
cell_width = (BOTTOM_RIGHT[0] - TOP_LEFT[0]) / GRID_WIDTH
cell_height = (BOTTOM_RIGHT[1] - TOP_LEFT[1]) / GRID_HEIGHT

# Tracking state variables (global for simplicity as per request, could be refactored into a class)
previous_grid = None
current_falling_piece = None
next_queue = []
first_piece_placed = False

# Tetromino shapes and their rotations (from tetris-move-solver.py)
# Each piece is represented as a list of 2D numpy arrays.
# '1' indicates a filled block, '0' is empty.
# The 'anchor' point (0,0) for each piece is typically its top-leftmost occupied cell
# in its initial orientation.
TETROMINOS = {
    'I': [
        np.array([[1, 1, 1, 1]]),
        np.array([[1], [1], [1], [1]])
    ],
    'O': [
        np.array([[1, 1], [1, 1]])
    ],
    'T': [
        np.array([[0, 1, 0], [1, 1, 1]]),
        np.array([[1, 0], [1, 1], [1, 0]]),
        np.array([[1, 1, 1], [0, 1, 0]]),
        np.array([[0, 1], [1, 1], [0, 1]])
    ],
    'S': [
        np.array([[0, 1, 1], [1, 1, 0]]),
        np.array([[1, 0], [1, 1], [0, 1]])
    ],
    'Z': [
        np.array([[1, 1, 0], [0, 1, 1]]),
        np.array([[0, 1], [1, 1], [1, 0]])
    ],
    'J': [
        np.array([[1, 0, 0], [1, 1, 1]]),
        np.array([[1, 1], [1, 0], [1, 0]]),
        np.array([[0, 0, 1], [1, 1, 1]]),
        np.array([[0, 1], [0, 1], [1, 1]])
    ],
    'L': [
        np.array([[0, 0, 1], [1, 1, 1]]),
        np.array([[1, 0], [1, 0], [1, 1]]),
        np.array([[1, 1, 1], [1, 0, 0]]),
        np.array([[1, 1], [0, 1], [0, 1]])
    ]
}


# --- Helper Functions (from main.py) ---

def get_cell_center(col, row):
    """Returns pixel coordinates of the center of a grid cell."""
    # Add TOP_LEFT coordinates to get absolute screen coordinates
    x = int(TOP_LEFT[0] + (col + 0.5) * cell_width)
    y = int(TOP_LEFT[1] + (row + 0.5) * cell_height)
    return x, y


def is_cell_filled(image, col, row, tolerance=50):
    """
    Checks if a cell at given grid (col, row) in the cropped image is filled by
    comparing its color to white.
    Image is expected to be the cropped grid image only, not full screenshot.
    """
    # Calculate pixel coordinates relative to the cropped grid image
    # Note: x and y here are relative to the top-left of the 'grid_img'
    x = int((col + 0.5) * cell_width)
    y = int((row + 0.5) * cell_height)

    img_array = np.array(image)

    # Ensure pixel coordinates are within the image bounds
    if not (0 <= y < img_array.shape[0] and 0 <= x < img_array.shape[1]):
        return False

    pixel = tuple(img_array[y, x][:3]) if img_array.ndim == 3 else (img_array[y, x],) * 3
    # Calculate the sum of absolute differences for RGB channels
    diff = sum(abs(int(p) - int(w)) for p, w in zip(pixel, WHITE))
    return diff > tolerance  # If difference is significant, it's not white (i.e., filled)


def analyze_grid(image):
    """
    Analyzes the entire Tetris grid from a cropped image and returns a 2D boolean array.
    True for filled, False for empty.
    """
    return np.array([[is_cell_filled(image, c, r) for c in range(GRID_WIDTH)] for r in range(GRID_HEIGHT)], dtype=bool)


def print_grid(grid):
    """Prints a visual representation of the grid using symbols."""
    if not isinstance(grid, np.ndarray):
        grid = np.array(grid)  # Ensure it's a numpy array for consistent behavior

    for r, row in enumerate(grid):
        print("".join("⬜" if cell else "⬛" for cell in row) + f" {r:02d}")
    print("0️⃣1️⃣2️⃣3️⃣4️⃣5️⃣6️⃣7️⃣8️⃣9️⃣")


def get_average_color(image, tolerance=30):
    """
    Returns the average color of non-white pixels in the image.
    Used for identifying next pieces.
    """
    img_array = np.array(image)
    if img_array.ndim == 3:
        # Create a mask for non-white pixels
        mask = np.sum(np.abs(img_array - WHITE), axis=2) > tolerance
        if np.any(mask):  # If there are any non-white pixels
            pixels = img_array[mask]
            return tuple(np.mean(pixels, axis=0).astype(int))
    return None


def identify_piece(avg_color, tolerance=50):
    """
    Matches an average color to a known Tetris piece type.
    Returns the piece name (e.g., 'T', 'I') or '?' if no match.
    """
    if avg_color is None:
        return "?"
    min_dist = float('inf')
    closest = "?"
    for name, color in PIECE_COLORS.items():
        # Calculate Euclidean distance between colors
        dist = sum((a - b) ** 2 for a, b in zip(avg_color, color)) ** 0.5
        if dist < min_dist:
            min_dist = dist
            closest = name
    # Only return closest if the distance is within tolerance
    return closest if min_dist < tolerance else "?"


def get_next_pieces(screenshot):
    """
    Returns a list of next three upcoming pieces from the preview window
    by analyzing the specified regions in the full screenshot.
    """
    return [
        identify_piece(get_average_color(screenshot.crop((*tl, *br))))
        for tl, br in NEXT_REGIONS.values()
    ]


def take_screenshot():
    """
    Captures the entire screen, crops the grid area, and identifies next pieces.
    Returns the analyzed grid (boolean array) and the list of next pieces.
    """
    screenshot = ImageGrab.grab()
    grid_img = screenshot.crop((*TOP_LEFT, *BOTTOM_RIGHT))
    return analyze_grid(grid_img), get_next_pieces(screenshot)


# --- Best Move Computation Functions (from tetris-move-solver.py) ---

def is_valid_position(grid, piece, row, col):
    """
    Checks if a piece can be placed at a given row, col on the grid without collision
    or going out of bounds. Returns True if valid, False otherwise.
    """
    piece_height, piece_width = piece.shape

    for r in range(piece_height):
        for c in range(piece_width):
            if piece[r, c] == 1:  # If this part of the piece is solid
                grid_row = row + r
                grid_col = col + c

                # Check bounds: piece must be within grid width and height
                if not (0 <= grid_col < GRID_WIDTH and 0 <= grid_row < GRID_HEIGHT):
                    return False
                # Check for collision with existing blocks on the grid
                if grid[grid_row, grid_col] == True:
                    return False
    return True


def hard_drop(grid, piece, start_row, start_col):
    """
    Simulates a hard drop for a piece from a starting position.
    Returns the new grid state after the piece has landed and the final row where it landed.
    """
    current_row = start_row
    # Move the piece down one row at a time until it's no longer a valid position
    while is_valid_position(grid, piece, current_row + 1, start_col):
        current_row += 1

    # Place the piece at its final resting position on a copy of the grid
    new_grid = np.copy(grid)
    piece_height, piece_width = piece.shape
    for r in range(piece_height):
        for c in range(piece_width):
            if piece[r, c] == 1:
                new_grid[current_row + r, start_col + c] = True
    return new_grid, current_row


def clear_lines(grid):
    """
    Clears any completed lines from the grid.
    Returns the new grid state and the number of lines cleared.
    """
    new_grid = np.copy(grid)
    lines_cleared = 0
    rows_to_keep = []  # Stores rows that are not full

    for r in range(GRID_HEIGHT):
        if not np.all(new_grid[r]):  # If the row is not completely full (contains at least one False)
            rows_to_keep.append(new_grid[r])
        else:  # If the row is full
            lines_cleared += 1

    # Create new empty rows to fill the top for the cleared lines
    empty_rows = np.full((lines_cleared, GRID_WIDTH), False)
    # Stack the empty rows on top of the rows that were kept
    final_grid = np.vstack((empty_rows, np.array(rows_to_keep)))

    return final_grid, lines_cleared


def evaluate_board(grid, lines_cleared):
    """
    Evaluates a given board state after a piece has landed and lines are cleared.
    Assigns a score based on various factors.
    """
    score = 0

    # 1. Reward for clearing lines: Highly desirable, so given a significant positive weight.
    score += lines_cleared * 100

    # 2. Penalize for aggregate height: Sum of the heights of all columns. Lower is better.
    aggregate_height = 0
    for col in range(GRID_WIDTH):
        # Find the highest filled cell (lowest row index) in the column
        for row in range(GRID_HEIGHT):
            if grid[row, col]:
                aggregate_height += (GRID_HEIGHT - row)  # Height from the bottom (GRID_HEIGHT-1)
                break  # Move to the next column once the highest block is found
    score -= aggregate_height * 2  # Arbitrary negative weight to penalize height

    # 3. Penalize for holes: Empty cells that have a filled cell directly above them.
    num_holes = 0
    for col in range(GRID_WIDTH):
        has_block_above = False
        for row in range(GRID_HEIGHT):
            if grid[row, col]:  # If there's a block
                has_block_above = True
            elif not grid[row, col] and has_block_above:  # If it's empty AND there was a block above
                num_holes += 1
    score -= num_holes * 50  # Arbitrary negative weight to penalize holes

    # Add bumpiness: Sum of absolute differences in height between adjacent columns
    bumpiness = 0
    column_heights = []
    for col in range(GRID_WIDTH):
        height = 0
        for row in range(GRID_HEIGHT):
            if grid[row, col]:
                height = GRID_HEIGHT - row  # Height from the bottom
                break
        column_heights.append(height)

    for i in range(GRID_WIDTH - 1):
        bumpiness += abs(column_heights[i] - column_heights[i + 1])
    score -= bumpiness * 5  # Arbitrary negative weight for bumpiness

    return score


def get_best_move(current_grid, piece_type):
    """
    Computes the best move (column + rotation) for a falling tetromino by
    simulating all possible drops and evaluating the resulting board states.
    """
    best_score = -float('inf')  # Initialize with a very low score
    best_move = None  # Stores (column, rotation_index)

    # Iterate through all possible rotations for the current piece
    for rotation_index, piece_shape in enumerate(TETROMINOS[piece_type]):
        piece_height, piece_width = piece_shape.shape

        # Iterate through all possible horizontal starting positions (columns)
        # The piece's leftmost point is 'col', its rightmost is 'col + piece_width - 1'.
        # Ensure the entire piece fits within the grid width.
        for col in range(GRID_WIDTH - piece_width + 1):
            # Pieces typically spawn at the very top (row 0 or slightly off-screen at -1)
            # For simulation, we start at row 0.
            start_row = 0

            # It's possible for a piece to not fit in certain positions even at spawn,
            # especially if the top of the grid is already filled.
            if not is_valid_position(current_grid, piece_shape, start_row, col):
                continue  # Skip this move if the piece can't even be placed initially

            # Simulate the hard drop to find where the piece would land
            simulated_grid, final_row = hard_drop(current_grid, piece_shape, start_row, col)

            # Clear any lines that are completed after the piece lands
            grid_after_clear, lines_cleared = clear_lines(simulated_grid)

            # Evaluate the resulting board state to get a score
            current_score = evaluate_board(grid_after_clear, lines_cleared)

            # If this move yields a better score, update our best move
            if current_score > best_score:
                best_score = current_score
                best_move = (col, rotation_index)

    return best_move


# --- Bot Action / Input Simulation ---

def execute_move(current_piece_type, target_col, target_rotation_index):
    """
    Executes the calculated best move using pydirectinput.
    Assumes the piece spawns at center top (col 3 or 4) and 0 rotations.
    """
    # Define keyboard keys for Tetris actions
    ROTATE_KEY = 'up'
    LEFT_KEY = 'left'
    RIGHT_KEY = 'right'
    HARD_DROP_KEY = 'space'

    # Current rotation index on screen (assuming 0 initially)
    # The actual piece_shape determines the number of rotations.
    current_rotation = 0  # Assume the piece spawns in its 0th rotation
    # Assuming initial spawn column is typically around 3 or 4 for most pieces.
    # We need to calculate the *relative* movement.
    # For simplicity, let's assume the piece is already at a column that allows movement to target_col.
    # A more robust bot would track current piece position on screen.
    # For now, let's assume the initial column for new pieces is always 3.
    # This is a simplification and may need adjustment based on your game.
    initial_spawn_col = SPAWN_POSITIONS[current_piece_type][target_rotation_index]

    print(f"Executing move: Target Column={target_col}, Target Rotation={target_rotation_index}")

    # 1. Perform Rotations
    num_rotations_needed = (target_rotation_index - current_rotation + len(TETROMINOS[current_piece_type])) % len(
        TETROMINOS[current_piece_type])
    for _ in range(num_rotations_needed):
        pydirectinput.press(ROTATE_KEY)
        time.sleep(0.01)  # Small delay to ensure input is registered

    # 2. Perform Horizontal Movement
    col_difference = target_col - initial_spawn_col
    if col_difference > 0:
        for _ in range(col_difference):
            pydirectinput.press(RIGHT_KEY)
            time.sleep(0.01)
    elif col_difference < 0:
        for _ in range(abs(col_difference)):
            pydirectinput.press(LEFT_KEY)
            time.sleep(0.01)

    # 3. Hard Drop
    pydirectinput.press(HARD_DROP_KEY)
    time.sleep(0.01)  # Wait a bit for the piece to settle


# --- Main Bot Logic ---

def track_and_play():
    """
    Main loop for tracking the game state and playing Tetris automatically.
    """
    global previous_grid, current_falling_piece, next_queue, first_piece_placed
    print("🎮 Starting Tetris auto-player... (First piece is skipped for initial setup)")

    while True:
        try:
            grid, next_pieces = take_screenshot()

            # Initialize the next_queue if it's empty
            if not next_queue:
                next_queue = next_pieces.copy()
                print(f"📦 Initial next queue: {next_queue}")
                # We skip the very first piece to ensure the queue is populated
                continue
            # Detect if a new piece has appeared by checking the next_queue
            # This indicates the previous piece has landed.
            elif next_queue != next_pieces:
                # The piece that was at the front of the queue is now the falling piece
                falling_piece = next_queue[0]
                next_queue = next_pieces.copy()  # Update the queue

                if first_piece_placed:
                    current_falling_piece = falling_piece
                    print(f"\n🔄 Next queue changed: {next_queue}")
                    print(f"🎯 Falling piece: {current_falling_piece}")
                    print_grid(grid)  # Print the grid before making a move decision

                    if current_falling_piece != '?':  # Only try to play if piece is identified
                        # Compute the best move for the current falling piece
                        best_move = get_best_move(grid, current_falling_piece)

                        if best_move:
                            best_col, best_rotation_index = best_move
                            print(f"💡 Best move found: Column {best_col}, Rotation {best_rotation_index}")
                            execute_move(current_falling_piece, best_col, best_rotation_index)
                            print("✅ Move executed.")
                        else:
                            print("⚠️ No valid move found for current piece. Skipping.")
                    else:
                        print("❓ Could not identify falling piece. Skipping move calculation.")

                    print("--------------------------------------------------")
                    # Small delay after executing a move to allow game state to update
                    time.sleep(0.05)
                else:
                    # Mark that the first piece has been handled (skipped)
                    first_piece_placed = True
                    print("🟢 First piece placed. Now tracking and playing...")

            time.sleep(0.01)  # Short delay to prevent excessive CPU usage from screenshots

        except KeyboardInterrupt:
            print("\n⏹️ Bot stopped by user.")
            break
        except Exception as e:
            print(f"❌ Error during tracking or playing: {e}")
            # Consider adding more specific error handling or logging
            time.sleep(0.1)  # Wait a bit before retrying after an error


if __name__ == "__main__":
    # Ensure pydirectinput is set up to not pause for failsafe
    pydirectinput.FAILSAFE = False
    pydirectinput.PAUSE = 0.01  # Small pause after each key press for consistency

    track_and_play()
