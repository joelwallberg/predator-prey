# Predator Prey Cellular Automaton
#
# Project in SI1336 by Joel Wållberg December 2021
#
#
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


class Site:
    """ Node class for keeping track of node data """

    def __init__(self, species):
        """ Constructor """
        self.species = species
        self.state = 'a'
        self.intention = None

class Tree(Site):
    """ Tree class """
    def __init__(self):
        super().__init__(species=-1)


class Grid:
    """ Grid class for keeping track of all nodes """
    def __init__(self, **kwargs):
        self.size = kwargs['size'] # grid size in x,y
        self.n_spec = kwargs['n_spec'] # num of species
        self.density = kwargs.get('density') # sum of initial densities of species can not exceed 1
        self.trees = kwargs.get('trees', False) # check if trees is active

        if (type(self.density) == float or type(self.density) == int) and not self.trees:
            self.density = self.density*np.ones(self.n_spec+1)
            self.density[0] = 0 # set tree density to zero
        elif self.density is None:
            self.density = np.zeros(self.n_spec+1)
        elif len(self.density) == self.n_spec and not self.trees:
                self.density = (0,) + self.density # set tree to zero if no given



        if self.density is not None:
            assert np.sum(self.density) <= 1

        self.grid = np.zeros(self.size, dtype=Site)
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                self.grid[i,j] = Site(0)

        self.dp = kwargs['dprob']
        self.bp = kwargs['bprob']
        if type(kwargs['r']) == tuple:
            self.r = kwargs['r']
        else:
            self.r = kwargs['r']*np.ones(self.n_spec)

        self.species_to_evaluate = kwargs.get('evaluate')
        self.setup(type=kwargs['setup'])



        #self.trees = self.distribute_trees(kwargs.p) # boolean array of trees
    def setup(self, type):
        """ Setup grid """
        if type not in ['uniform','5050','303030', 'evaluate', 'evaluate_trees']:
            print("Unrecognized setup. Uniform setup is used")
            type = 'uniform'

        if type == 'uniform': #or type == 'evaluate_trees':

            temp= []
            if self.trees:
                for _ in range(int(self.density[0]*self.size[0]*self.size[1])):
                    temp.append(Tree())
            for species in range(1,self.n_spec+1):
                for _ in range(int(self.density[species]*self.size[0]*self.size[1])): # add correct density of each species
                    temp.append(Site(species=species))

            for _ in range(int(self.size[0]*self.size[1] - len(temp))): # set all remaining sites to empty
                temp.append(Site(species=0))

            np.random.shuffle(temp) # shuffle all species
            self.grid = np.array(temp).reshape(self.size[0], self.size[1])
            #if type=='evaluate_trees':
            #    self.grid[self.size[0]//2,:] = Tree()

        if type == 'evaluate_trees':
            mx,my = self.size[0]//2, self.size[1]//2
            self.grid[:,my] = Tree()
            self.grid[mx,my-1] = Site(3)
            self.grid[mx,my-2] = Site(2)
            self.grid[mx,my+1] = Site(2)

        if type == 'evaluate':
            for i in range(self.size[0]):
                for j in range(self.size[1]):
                    if self.species_to_evaluate == 1 and (i,j)==(1,1):
                        self.grid[i,j] = Site(species=1)
                    elif self.species_to_evaluate != 1:
                        self.grid[i,j] = Site(species=self.species_to_evaluate)
                    else:
                        self.grid[i,j] = Site(species=0)


        #if type == 'evaluate_trees':
        #    x0, y0 = self.size[0]//2, self.size[1]//2
        #    self.grid[x0,:] = Tree()
        #    self.grid[x0-1,:] = Site(species=1)
        #    self.grid[x0+1,:] = Site(species=2)

        if type == '5050':
                for i in range(self.size[0]):
                    for j in range(self.size[1]//2):
                        self.grid[i,j] = Site(species=1)
                        self.grid[i,j+self.size[1]//2] = Site(species=2)
        if type == '303030':
                for i in range(self.size[0]):
                    for j in range(self.size[1]//3):
                        self.grid[i,j] = Site(species=1)
                        self.grid[i,j+self.size[1]//3] = Site(species=2)
                        self.grid[i,j+2*self.size[1]//3] = Site(species=3)

    def step(self, improved=False):
        """ Take a simulation step """
        self._attack(improved=improved)
        self._reprod()
        self._move()


    def vonNeumann(self, idx, count=None, state=None, returnNeighs=False):
        """ Generate von Neumann neighborhood of manhattan dist. 1 """
        x,y = idx
        mx = self.size[0]
        my = self.size[1]
        neighs = [self.grid[x,(y-1)%my], self.grid[(x+1)%mx,y],
                self.grid[x,(y+1)%my], self.grid[(x-1)%mx,y]]
        if returnNeighs:
            return neighs
        n = 0

        for neigh in neighs:
            if neigh.species == count:
                if state is None:
                    n += 1
                elif neigh.state == state:
                    n += 1

        return n

    def moore(self, idx, r, count, state=None):
        """ Generate Moore neighborhood of radius r """
        x,y = idx
        mx = self.size[0]
        my = self.size[1]
        neighs = []
        for dx in range(-r,r+1):
            for dy in range(-r,r+1):
                if dx != 0 and dy != 0:
                    neighs.append(self.grid[(x+dx)%mx,(y-dy)%my])

        n = 0

        for neigh in neighs:
            if neigh.species == count:
                if state is None:
                    n += 1
                elif neigh.state == state:
                    n += 1

        return n

    def f(self,v):
        if v==1:
            return 1-np.cos(np.pi/2 * self.bp)
        elif v==2:
            return 1-np.exp(-np.exp(1)*self.bp)

    def _attack(self, improved=False):
        """ Attack phase """
        for i,row in enumerate(self.grid):
            for j,site in enumerate(row):

                if site.species in (-1,0):
                    continue

                if site.species != self.n_spec and site.species != 0:

                    n_pt = self.vonNeumann(idx=(i,j), count=site.species+1)
                    if improved:
                        r = 1 # Moore radius
                        n_pr = self.moore(idx=(i,j),r=r, count=site.species+1)
                        if n_pt > 0 or n_pr == 0:
                            if np.random.rand() < n_pr*self.f(v=2)/(2*r+1)**2:
                                site.species=0
                                site.state='b'
                    else:
                        if np.random.rand() > (1-self.dp[site.species-1])**n_pt:
                            site.species = 0 # died
                            site.state = 'b'


                if site.species != 1 and site.species != 0:
                    n_pr = self.vonNeumann(idx=(i,j), count=site.species-1)

                    if np.random.rand() < (1-self.dp[site.species-2])**n_pr:
                        site.state = 'a' # failed hunt
                    else:
                        site.state = 'b' # eaten

    def _reprod(self):
        """ Reproduction phase """
        for i,row in enumerate(self.grid):
            for j,site in enumerate(row):

                if site.species == -1:
                    continue

                if site.species == 1:
                    continue # do nothing

                elif site.species > 1:
                    if np.random.rand() < self.dp[site.species-1]:
                        site.species = 0 # natural death


                elif site.species == 0:
                    new = []
                    if site.state == 'a': # if empty without kill
                        preys = self.vonNeumann(idx=(i,j),count=1)
                        preds = self.vonNeumann(idx=(i,j),count=2)
                        if preys == 0 or preds != 0:
                            continue
                        elif np.random.rand() < 1-(1-self.bp[0])**(preys):
                            new.append(1)

                    elif site.state == 'b': # if empty due to kill
                        for species in range(1,self.n_spec):
                            preds_eaten = self.vonNeumann(idx=(i,j),count=species+1, state='b')
                            #print("Preds_eaten=", preds_eaten)

                            if np.random.rand() < 1-(1-self.bp[species])**preds_eaten:
                                new.append(species+1)


                    try:
                        site.species = np.random.choice(new)
                    except:
                        site.species = 0


        for i,row in enumerate(self.grid):
            for j,site in enumerate(row):
                site.state = 'a' # set hungry for next step

    def _move(self):
        """ Movement phase """
        intentions = np.zeros(self.size)
        for i,row in enumerate(self.grid):
            for j,site in enumerate(row):
                #print(i,j, '   ', site.species)
                self.set_intention(site, i,j)

        for i,row in enumerate(self.grid):
            for j,site in enumerate(row):
                if site.species == 0: # if empty, search von Neumann neigh. for incoming species
                    incoming = []
                    N,E,S,W = self.vonNeumann(idx=(i,j), returnNeighs=True)
                    if N.intention == (0,1) and N.species != 0:
                        incoming.append(N)
                    if E.intention == (-1,0) and E.species != 0:
                        incoming.append(E)
                    if S.intention == (0,-1) and S.species != 0:
                        incoming.append(S)
                    if W.intention == (1,0) and W.species != 0:
                        incoming.append(W)

                    if len(incoming) > 0: # choose random incoming
                        incoming = np.random.choice(incoming)
                        site.species = incoming.species
                        incoming.species = 0
                    else:
                        continue
                    #print('Site=',site.species, 'Incoming=', incoming.species)





    def set_intention(self, site, x, y):
        """ Determine intention of movement """

        if site.species in (-1,0):
            return

        elif site.species == 1:
            directions = self._mooreQuadrants(2, x, y, r=self.r[0])
            dir = np.where(directions == directions.min())[0]

            if directions.max() == 0: # if no preds in neigh, stand still

                site.intention = None
                return

        elif site.species > 1 and site.species < self.n_spec:
            pred_dir = self._mooreQuadrants(site.species+1, x, y, r=self.r[site.species-1])
            prey_dir = self._mooreQuadrants(site.species-1, x, y, r=self.r[site.species-1])

            pd_dir = np.where(pred_dir == pred_dir.min())[0]
            py_dir = np.where(prey_dir == prey_dir.max())[0]

            if pred_dir.max() == 0 and prey_dir.max() == 0:
                dir = [np.random.choice([0,1,2,3])] # if no itention go random
            elif prey_dir.max() > pred_dir.min():
                dir = py_dir
            else:
                dir = pd_dir

        elif site.species == self.n_spec:
            directions = self._mooreQuadrants(self.n_spec-1, x, y, r=self.r[self.n_spec-1])
            dir = np.where(directions == directions.max())[0]


        #if len(dir) > 1: # if multiple directions choose one random
        dir = np.random.choice(dir)

        if dir == 0: # North
            site.intention = (0,-1) # north is a neg. step in the y direction
        elif dir == 1: # East
            site.intention = (1,0) # and south is a pos. step in the y direction
        elif dir == 2: # South
            site.intention = (0,1)
        elif dir == 3: # West
            site.intention = (-1,0)


    def _mooreQuadrants(self, v, x, y, r=3):
        """ Calculate Moore quadrant number """

        def quadrant(dx,dy,i):
            """ Calculate one quadrant row """
            neigh = self.grid[(x+dx)%mx, (y+dy)%my]
            if type(neigh) == Tree:
                trees.add((dx/i,dy/i))

            elif neigh.species == v and (dx/i, dy/i) not in trees:
                return 1
            return 0

        mx,my = self.size

        counter = [0,0,0,0]
        trees = set([])
        for i in range(1,int(r+1)):
            for j in range(-i,i+1):
                #dx, dy = j, -i
                counter[0] += quadrant(j,-i,i)
                counter[1] += quadrant(i,j,i)
                counter[2] += quadrant(j,i,i)
                counter[3] += quadrant(-i,j,i)

                # The old without trees
                #if self.grid[(x+j)%mx, (y-i)%my].species == v:
                #    nN += 1
                #if self.grid[(x+i)%mx, (y+j)%my].species == v:
                #    nE += 1
                #if self.grid[(x+j)%mx, (y+i)%my].species == v:
                #    nS += 1
                #if self.grid[(x-i)%mx, (y+j)%my].species == v:
                #    nW += 1

        return np.array(counter)


    def get_species(self):
        species = np.zeros(self.size)
        for i,row in enumerate(self.grid):
            species[i,:] = [site.species for site in row]
        return species

    def get_phases(self):
        phases = np.zeros(self.n_spec)
        for i,row in enumerate(self.grid):
            for site in row:
                if site.species not in (-1, 0):
                    phases[site.species-1] += 1

        phases = phases/((1-self.density[0])*self.size[0]*self.size[1]) # divide by number of available positions
        return phases


    def show(self, cmap='Greys'):
        """ Show grid """
        plt.figure()
        plt.imshow(self.get_species(), cmap=cmap)
        plt.axis('off')
        #plt.colorbar()
        #plt.show()



def animate(frame, grid, im):
    grid.step()
    spec = grid.get_species()
    im.set_array(spec)
    return im,

class Simulation:

    def __init__(self, **kwargs):
        self.CA = Grid(**kwargs)
        self.kwargs = kwargs
        self.phase_density = []

    def run(self,steps):
        self.phase_density = np.zeros((self.kwargs['n_spec'],steps))
        for i in range(steps):
            self.CA.step()
            self.phase_density[:, i] = self.CA.get_phases()

    def run_animate(self, steps=25):
        plt.figure()
        im = plt.imshow(self.CA.get_species(), vmin=-1, vmax=self.kwargs['n_spec'], cmap='Greys')
        plt.colorbar()

        anim = animation.FuncAnimation(plt.gcf(), animate,
                                      fargs=[self.CA,im], frames=steps, interval=1, blit=False, repeat=False)
        plt.show()

    def population(self):
        return self.CA.get_phases()

    def plot_populations(self, save=False, format='.pdf'):
        """ Plot relative population sizes vs. time """
        plt.figure()
        for i,species in enumerate(self.phase_density):
            plt.plot(species, label='Species {}'.format(i+1))
        plt.legend()
        plt.xlabel('time')
        plt.ylabel('population')

        if save: plt.savefig(save+format)


    def plot_phase_portrait(self, save=False, format='.pdf'):
        """ Plot phase portrait """

        if len(self.phase_density) == 3:
            fig = plt.figure()
            ax = plt.axes(projection='3d')
            ax.scatter3D(self.phase_density[0],self.phase_density[1],self.phase_density[2], c='black', s=5)
            plt.title('3D Phase portrait\nsteps = {}'.format(len(self.phase_density[0])))
            ax.set_xlabel('Species 1')
            ax.set_ylabel('Species 2')
            ax.set_zlabel('Species 3')


        for i in range(1,len(self.phase_density)):
            try:
                x,y = self.phase_density[i-1],self.phase_density[i]
                plt.figure()
                ymin,ymax = y.min(),y.max()
                plt.scatter(x,y, c='black',s=5)
                plt.xlabel('Species {}'.format(i))
                plt.ylabel('Species {}'.format(i+1))


                plt.title('Phase portrait')
                if save: plt.savefig(save+"_{}_{}".format(i,i+1)+format)
            except:
                break






def main():
    np.random.seed(1)
    
    #BPROB = (.9, .6, .5) # the ones used in the report
    #DPROB = (.9, .2, .01)
    BPROB=(.1,.7,.88) # more movement with these probabilities
    DPROB=(.95,.2,.01)

    CA = Simulation(size=(124,124), n_spec=3, trees=True, density=(.1,.05,.05,.05), bprob=BPROB, dprob=DPROB, r=1, setup='uniform', evaluate=2)
    CA.run_animate(steps=1000)

    #CA.run(steps=200)
    #CA.CA.show()
    #CA.plot_populations()
    #CA.plot_phase_portrait()
    plt.show()



if __name__ == '__main__':
    main()
