import numpy as np
import matplotlib.pyplot as plt

piece_base = np.array([[0,0,0],[1,0,0],[2,0,0],[3,0,0],[2,1,0]])

rot_x = np.array([[1,0,0],[0,0,-1],[0,1,0]])
rot_y = np.array([[0,0,1],[0,1,0],[-1,0,0]])
rot_z = np.array([[0,-1,0],[1,0,0],[0,0,1]])

class Modele():
    cube = np.zeros((5,5,5))
    pieces = []
    def __init__(self):
        self.cube = np.zeros((5,5,5))
        self.pieces = []
    
    def add_piece(self, r, piece):
        self.pieces.append(piece)
        for i in range(5):
            self.cube[r[i][0]][int(r[i][1])][int(r[i][2])] = 1
        


class Piece():
    c_rotx : int = 0
    c_roty : int = 0
    c_rotz : int = 0
    c_trans : np.array = np.array([0,0,0])

    def __init__(self, c_rotx, c_roty, c_rotz, c_trans):
        self.c_rotx = c_rotx
        self.c_roty = c_roty
        self.c_rotz = c_rotz
        self.c_trans= np.stack([c_trans for i in range(5)])

    def is_valid(self,r, cube):
        for i in range(5):
            if r[i][0] < 0 or r[i][0] > 4 or r[i][1] < 0 or r[i][1] > 4 or r[i][2] < 0 or r[i][2] > 4:
                return False
            
            if cube[int(r[i][0])][int(r[i][1])][int(r[i][2])] == 1:
                return False
        return True
    
    def repr(self):
        return piece_base @ np.linalg.matrix_power(rot_x, self.c_rotx) @ np.linalg.matrix_power(rot_y, self.c_roty) @ np.linalg.matrix_power(rot_z, self.c_rotz) + self.c_trans

def parcours(modele, piece):
    r = piece.repr()
    if np.random.random() < 0.000001:
        print(modele.cube)
    if len(modele.pieces) == 25:
        print("done")
        print(modele.pieces)
        print(modele.cube)
        return True

    if piece.is_valid(r, modele.cube):
        modele.add_piece(r, piece)
        for i in range(4):
            for j in range(3):
                for k in range(2):
                    for t in [np.array([l,m,n]) for l in range(5) for m in range(5) for n in range(5)]:
                        if parcours(modele, Piece(i,j,k,t)):
                            return True
        for i in range(5):
            modele.cube[int(r[i][0])][int(r[i][1])][int(r[i][2])] = 0
        modele.pieces.pop()

    return False

modele = Modele()
parcours(modele, Piece(0,0,0,np.array([0,0,0])))

print(len(modele.pieces))
print(modele.pieces)

    