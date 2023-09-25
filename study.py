# PARAMETER STUDY
#
#   Parameters that are studied:
#       - effect of lattice sizes
#       - effect of initial densities
#       - length of sight, i.e. Moore radius r of each creature
#
#

from predprey import *
import multiprocessing as mp
from itertools import permutations

SIZE = (64,64)
BPROB = (.9, .6, .5)
DPROB = (.9, .2, .01)


R = 1
T = 0 # tree density
DENSITY = (T,.05,.05,.05)
STEPS = 300


def run_simulation(x, type, plot=False, savefig=False, animate=False, plot_phases=False, plot_populations=False, multistudy=False):
    """ Run one simulation with parameter x of type type """
    steps = STEPS
    if type == 'r':
        CA = Simulation(size=SIZE, n_spec=3, density=DENSITY, bprob=BPROB, dprob=DPROB, r=x, setup='uniform')
    elif type == 'd':
        CA = Simulation(size=SIZE, n_spec=3, trees=True, density=x, bprob=BPROB, dprob=DPROB, r=R, setup='uniform')
    elif type == 's':
        CA = Simulation(size=(x,x), n_spec=3, density=DENSITY, bprob=BPROB, dprob=DPROB, r=R, setup='uniform')
    elif type == 't':
        density = (x,) + DENSITY[1:]
        CA = Simulation(size=SIZE, n_spec=3, trees = True, density=density, bprob=BPROB, dprob=DPROB, r=R, setup='uniform')
    elif type == 'endtime':
        CA = Simulation(size=SIZE, n_spec=3, density=DENSITY, bprob=BPROB, dprob=DPROB, r=R, setup='uniform')
        steps = x

    if animate:
        CA.run_animate(steps=steps)
    else:
        CA.run(steps=steps)
        if multistudy:
            return CA.phase_density
        if plot_phases:
            d = CA.phase_density
            c,s = 'black',2
            fig = plt.figure(figsize=(6,3))
            ax = fig.add_subplot(1,2,1)
            ax.set_xlabel('Species 1')
            ax.set_ylabel('Species 2')
            ax.scatter(d[0,:],d[1,:],c=c,s=s)
            ax = fig.add_subplot(1,2,2)
            ax.scatter(d[1,:],d[2,:],c=c,s=s)
            ax.set_xlabel('Species 2')
            ax.set_ylabel('Species 3')
            plt.suptitle('Phase diagram, steps={}'.format(steps))
            plt.tight_layout()
            if savefig: plt.savefig(savefig)
            plt.show()

    p = CA.population()
    if plot_populations:
        CA.plot_populations()
        if savefig:
            plt.savefig(savefig)
        plt.show()
    if plot:
        CA.CA.show()
        if savefig:
            plt.savefig(savefig)
        plt.show()

    return p


def study(X, type, n=1, save=False):
    """ Study of parameter type on equilibrium populations.
        run_simulation() will run for each entry in X n times. """
    p = []

    for x in X:
        _config = []
        for i in range(n):
            print(type + ',n =', x, i)
            _config.append(run_simulation(x=x, type=type))
        p.append(_config)

    if save:
        np.save(save, p)
    return p

def _run(x,type):
    return run_simulation(x=x, type=type, multistudy=True)

def multistudy(type, **kwargs):
    """ Perform multistudy on all possible
        combinations of elements of X. """

    if type == 'd':
        densities = []
        p = kwargs['partitions']

        combs = int((p+1)*(p-1)*(p)/6)
        print('combinations = ', combs)

        for x in range(1, p+1):
            for y in range(1, p - x + 1):
                for z in range(0, p - (x + y) + 1):
                    density = (T,x/p,y/p,z/p)
                    density = [x/(1+T) for x in density]
                    densities.append(density)
        x = densities
    else:
        x = kwargs['params']


    nprocs = mp.cpu_count()
    pool = mp.Pool(processes=nprocs) # run multi process
    args = [(c,type) for c in x]
    res = pool.starmap(_run, args)
    res = np.array(res)
    print(res.shape)
    if kwargs.get('save', False):
        print('saved')
        np.save(kwargs['save'], res)

def plot_multi(file, type, **kwargs):

    res = np.load(file)
    fig, axs =  plt.subplots(1,3, figsize=(15,5),sharey=True, sharex=True)
    fig2, axs2 =  plt.subplots(1,2, figsize=(10,5), sharey=True, sharex=True)
    for i in range(3):
        for j in range(len(res)):

            if type == 'd':
                if res[j,-1,-1] > 0.1:
                    c = 'red'
                    a = 0.4
                    print(res[j,:,0])
                else:
                    c = 'black'
                    a = 0.2
                axs[i].plot(res[j,i,:], color=c, alpha=a, linewidth=1)

                if i < 2:
                    axs2[i].plot(res[j,i,:],res[j,i+1,:],color=c,linewidth=1)
                    axs2[i].set_xlabel('Species {}'.format(i+1))
                    axs2[i].set_ylabel('Species {}'.format(i+2))


            else:
                axs[i].plot(res[j,i,:], linewidth=1, label=type + ' = ' + str(kwargs['params'][j]))

        axs[i].set_title('Species {}'.format(i+1))

    if type != 'd':
        handles, labels = axs[-1].get_legend_handles_labels()
        fig.legend(handles, labels, loc=(0.4,0.8), prop={'size':15},ncol=len(axs[-1].lines))

    fig.supylabel('population')
    fig.supxlabel('time')
    plt.ylim([0,1])
    fig.tight_layout()
    fig2.tight_layout()

    savefig = kwargs.get('savefig', False)
    if savefig:
        format='.pdf'
        fig.savefig(savefig+format)
        fig2.savefig(savefig+'phase'+format)
    plt.show()

def plot_moving_average(files,savefig=False):
    fig, axs = plt.subplots(2,2, sharex=True, sharey=True)
    fig2,ax2 = plt.subplots()
    d = {0:(0,0),1:(0,1),2:(1,0),3:(1,1)}
    R = ['1','5','10','(10,5,1)']
    for i,f in enumerate(files):
        f += '.npy'
        data = np.load(f)
        data = np.array([x for x in data if x[-1,-1]>0.1]) # only keep data where all survive

        avg = np.zeros((data.shape[1], data.shape[2]))
        err = np.zeros((data.shape[1], data.shape[2]))
        for j in range(data.shape[2]):
            for k in range(3):
                avg[k, j] = np.mean(data[:,k,j])
                err[k, j] = np.std(data[:,k,j]) # standard error
        print(err.shape)
        for s in range(avg.shape[0]):
            markers, caps, bars= axs[d[i]].errorbar(x=[x for x in range(avg.shape[-1])],y=avg[s,:],yerr=err[s,:])
            [bar.set_alpha(0.1) for bar in bars]

        if i in (0,2):
            m = ':'
            if i == 0:
                m = '--'
            ax2.plot(err[0,10:], label=r'$r=$'+R[i], linestyle=m, color='black')
        axs[d[i]].set_title(r'$r=$'+R[i])
        axs[d[i]].set_ylim([0,1])

    fig.supxlabel('time')
    fig.supylabel('Running average population')

    fig.legend(['Species '+str(s+1) for s in range(3)],ncol=3, loc='upper center')


    fig2.supxlabel('time')
    fig2.supylabel('Running st.d.')
    ax2.legend()
    ax2.set_xlim([-10,300])
    if savefig:
        fig.savefig(savefig)
        fig2.savefig('figs/errorLOS'+savefig[-4:])
    plt.show()


def statistics(file, type, X=None, savefig=False):
    """ Perform statistics on data """
    data = np.load(file+'.npy', allow_pickle=True)
    mean = []
    err = []

    # calculate mean of each experiment
    for experiment in data:
        N = len(experiment)
        print(N)
        mean.append(np.array(experiment).mean(axis=0))
        err.append(np.array(experiment).std(axis=0)/np.sqrt(N))

    mean = np.array(mean)
    err = np.array(err)
    print('mean = \n', mean)
    print('error = \n', err)

    plt.figure()
    for i in range(3):
        plt.errorbar(X, mean[:,i], yerr=err[:,i], capsize=2, label="Species = {}".format(i+1))

    plt.xlabel(type)
    plt.ylabel('population')
    plt.legend()
    if savefig:
        plt.savefig(savefig)
    plt.show()








def main():
    global R,T
    np.random.seed(1)
    dir = 'figs/'
    format = '.pdf'


    """SNAPSHOT"""
    #run_simulation(x=1, type='r', plot=True, savefig=dir+'snapshot'+format)


    """END TIME STUDY"""
    #run_simulation(x=500, type='endtime', plot_populations=True)
    #np.random.seed(1)
    #run_simulation(x=100, type='endtime', plot_phases=True, savefig=dir+'endtime100'+format)
    #np.random.seed(1)
    #run_simulation(x=300, type='endtime', plot_phases=True, savefig=dir+'endtime300'+format)
    #np.random.seed(1)
    #run_simulation(x=500, type='endtime', plot_phases=True, savefig=dir+'endtime500'+format)


    """MULTISTUDY OF GRID SIZE"""
    #file = 'multi_gridsize'
    #S = [2**x for x in range(3,8)]
    #multistudy(type='s', params=S, save=file)
    #plot_multi(type='s', file=file+'.npy', params=S, savefig=dir+file+format)


    """MULTISTUDY of R = 1,5,10"""
    # for r in [1,5,10]:
    #     R = r
    #     file = 'multi_density_r' + str(r)
    #     multistudy(type='d', partitions=8, save=file)
    #     plot_multi(type='d', file=file+'.npy', savefig=dir+file)


    """MULTISTUDY OF ASYNCHRONOUS R"""
    # R = (10,5,1)
    # file = 'multi_density_async_10_5_1'
    # multistudy(type='d', partitions=8, save=file)
    # plot_multi(type='d', file=file+'.npy', savefig=dir+file)


    """PLOT MOVING AVERAGE AND DEVIATION OF R STUDY"""
    # files = ['multi_density_r'+str(r) for r in [1,5,10]] + ['multi_density_async_10_5_1']
    # plot_moving_average(files, savefig=dir+'running_avg_LOS'+format)


    """MULTISTUDY OF TREE DENSITY"""
    # R = 3
    # for t in [10,30,50,80]:
    #     t = t/100
    #     T = t/(1-t) # to get correct density
    #     file = 'multi_density_r3_t' + str(t)
    #     multistudy(type='d', partitions=8, save=file)
    #     plot_multi(type='d', file=file+'.npy', savefig=dir+file)






if __name__ == '__main__':
    main()
