CLASS = ("empty", "paper", "rock", "scissors") # empty must be index zero or else the code will break; must be alphabetically ordered bcs of how output neurons are arranged
BX, BY, BS = 70, 70, 192 # coords and size of box used to capture image data
WW, WH = 640, 480 # window dimensions (img will be scaled if cam res different)
FPS = 1000 // 10
ASSET_DIR = "assets" # for sounds etc
DATA_DIR = "data" # where to store captured images
DATA_SIZE = 48 # side of captured data in pixels
LOG_BUFFER_SIZE = 6
GESTURE_BUFFER_SIZE = 8

# convenients color constants
BLACK = (0,0,0)
WHITE = (255,255,255)
GRAY = (92,92,92)
RED = (0,0,255)
GREEN = (0,255,0)
YELLOW = (0,255,255)

# used in training the network
BATCH_SIZE = 32
EPOCHS = 15
