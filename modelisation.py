import numpy as np
import matplotlib.pyplot as plt

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

n = 300000
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
    model_1.step(epsilon = 0.00001/np.log(epoch+2), zeta = 0.01/np.log(epoch+2), nu = 0.0001/np.log(epoch+2))

print("end: ", model_1.show())
for i in range(25):
    print(model_1.pieces[i].piece)
    print("end loss: ", model_1.pieces[i].loss)
plt.plot(loss)
plt.show()

