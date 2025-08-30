# importo pacchetti
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from typing import Optional
from pathlib import Path

# Definisco una classe con i parametri della simulazione e le funzioni 
# che utilizzo in heartbeat per lo studio real time della frizione dinamica

class setup_class:




    def __init__(self, boxL,nx, ny, N, nrows, m_host, m_BH, Vx, r, tau, T):
        boxL, nx, ny, N, nrows, ncols, m_host, m_BH, x_BH, x0_BH, v_BH, Vx, xs, ys, r, tau = build_setup(boxL, nx, ny, N, nrows, m_host, m_BH, Vx, r, tau, T)

        self.boxL = boxL
        self.nx = nx
        self.ny = ny
        self.N = N
        self.nrows = len(ys)
        self.ncols = ncols

        self.m_host = m_host
        self.m_BH = m_BH
        self.x_BH = x_BH
        self.x0_BH = x0_BH
        self.v_BH = v_BH
        self.Vx = Vx

        self.xs = xs
        self.ys = ys
        self.r = r  
        self.tau = tau
        self.T = T

        self.slow_stars_fraction = 1

 
    def compute_deflection_angle(self, sim, particle_row, scan_folder, sub_folder, frame, savefig=False, subtitle = ''):

        """
        Nel logaritmo dell'eq di Chandrasekhar compare il parametro di impatto critico b90, associato a deviazioni di 90°. Questa funzione ne ottiene una stima.

        Anzitutto calcola di quanto vengono deviate le orbite delle particelle a seguito dell'interazione gravitazionale con il BH (interazione dominante). 
        Poichè viene calcolato l'angolo di deviazione di ciascuna particella presente nella box, si possono distinguere due popolazioni: le particelle che 
        devono ancora passare il BH (angoli di deviazione piccoli) e le particelle che lo hanno già passato (angoli di deviazione grandi). 

        Siccome l'angolo di deviazione dipende, tramite l'eq (3) in main.ipynb dal parametro d'impatto b (noto per costruzione della simulazione) e 
        dal parametro critico b90 (incognita) possiamo ricavare quest'ultimo eseguendo un fit in cui si considerano per ogni b iniziale l'angolo 
        di deviazione massimo (corrispondente a particelle prossime ad uscire dalla box). 

        Infine la funzione può creare un plot degli angoli di deviazione di tutte le particelle in funzione dei parametri di impatto iniziali, sovrapponendo
        il fit tramite l'eq (3) e gli angoli di deviazione estremi usati per il fit.

        Parameters
        ----------
        sim : class
            Classe costruita tramite rebound
            
        particle_row : int
            Il numero della riga in cui una particella è stata inserita nella box, a partire dal basso.

        scan_folder : string
            Cartella dello scan delle simulazioni, in cui viene salvato il file .pickle contentente tutte le run e il file di riepilogo .txt

        sub_folder : string
            Sottocartella in cui vengono salvate le immagini prodotte durante la singola run.

        frame : int
            Indice crescente che parte da 1 per distinguere in modo semplice le figure associate ad un valore di Vx

        savefig : bool
            Booleano per salvare o meno le figure

        subtitle : string
            Eventuale sottotitolo nelle figure
            
        Returns
        -------
        b90_fit : float
            Il valore del parametro critico ottenuto dal fit
        """


        def_angle_extremes = np.zeros(self.nrows)
        def_angles_plot = []
        ys_plot = []
        
        for row in range(0,self.nrows):
            def_angles = []
            for p in sim.particles:
                if p.hash.value != 1753590236  and particle_row[p.hash.value]==row:          # non considero il black hole, che ha hash 1753590236 
                    
                    teta = np.arctan2(p.vy,p.vx)                                    # direzione della stella
                    def_angle = (np.abs(teta)-np.pi)*np.sign(teta)*180/np.pi        # deviazione rispetto alla direzione di origine (destra)
                    def_angles = np.append(def_angles, def_angle)                   # salvo l'angolo in gradi

                    def_angles_plot = np.append(def_angles_plot, def_angle)
                    ys_plot = np.append(ys_plot,self.ys[row])

            if self.ys[row]<0:
                def_angle_extremes[row] = np.min(def_angles)
            else:
                def_angle_extremes[row] = np.max(def_angles)

        f_teta_def = lambda x,b90: 2*np.arctan(b90/x) *180/np.pi

        popt, _ = curve_fit(f=f_teta_def, xdata = self.ys, ydata = def_angle_extremes)        # calcolo b90_fit dal fit
        b90_fit = popt[0]
        x_fit = np.linspace(-self.boxL*self.ny/2*0.9,self.boxL*self.ny/2*0.9,200)
        y_fit = f_teta_def(x_fit, b90_fit)

        if savefig:

            # Crea sottocartella se non esiste
            Path(scan_folder + sub_folder).mkdir(parents=False, exist_ok=True)

            fig, ax = plt.subplots()
            ax.plot(ys_plot,def_angles_plot,'C0.',alpha=0.1)
            ax.plot(self.ys,def_angle_extremes,'rx',label='extremes')
            ax.plot(x_fit,y_fit,'k-',alpha=0.5,label=f'fit, b90_fit={b90_fit:.2f}')
            ax.set_title('Deflection angle vs impact parameter' + subtitle)
            ax.set_xlabel('Impact parameter')
            ax.set_ylabel('Deflection angle [deg]')
            ax.legend(loc='upper left')
            plt.savefig(scan_folder + sub_folder + rf'\da_{frame:d}.png')
            plt.close()

        return  b90_fit 




    def compute_velocity_dispersion(self, sim, scan_folder, sub_folder, frame, savefig=False, subtitle = ''):
        """
        Nell'eq di Chandrasekhar compare la frazione di stelle con velocità in modulo minore a quella del BH. Questa funzione la calcola.


        Nel farlo tiene conto del fatto che il numero di stelle nella box non è esattamente costante.
        Infine la funzione può plottare la distribuzione dei moduli delle velocità e delle componenti x e y, con una linea verticale a rappresentare 
        la velocità orizzontale del BH.


        Parameters
        ----------
        sim : class
            Classe costruita tramite rebound
            
        scan_folder : string
            Cartella dello scan delle simulazioni, in cui viene salvato il file .pickle contentente tutte le run e il file di riepilogo .txt

        sub_folder : string
            Sottocartella in cui vengono salvate le immagini prodotte durante la singola run.

        frame : int
            Indice crescente che parte da 1 per distinguere in modo semplice le figure associate ad un valore di Vx

        savefig : bool
            Booleano per salvare o meno le figure

        subtitle : string
            Eventuale sottotitolo nelle figure
            
 
 
        Returns
        -------
        b90_fit : float
            Il valore del parametro critico ottenuto dal fit
        """


        # Calcolo la frazione di particelle con una velocità assoluta inferiore al BH
        slower_stars = 0
        vs = []             # velocirtà delle stelle

        for p in sim.particles: 
            v = np.sqrt((p.vx + self.Vx)**2+p.vy**2)
            vs = np.append(vs,v) 
            if v < sim.particles['BH'].vx + self.Vx:                   # qui sto utilizzando la velocità iniziale del BH. Con v_squared < Vx**2 sembra buono
                                                                # oppure v_squared*m_host/N < Vx**2*m_BH
                slower_stars += 1

        self.slow_stars_fraction = slower_stars/sim.N

        pv = np.array([np.sqrt((p.vx + self.Vx)**2+p.vy**2) for p in sim.particles])
        pvx = np.array([p.vx + self.Vx for p in sim.particles])
        pvy = np.array([p.vy for p in sim.particles])

        pvx_std = pvx.std()
        pvy_std = pvy.std()
        pv_std = np.sqrt(pvx_std**2 + pvy_std**2)

        v_rms = np.mean(vs)     # stavo prendendo sqrt
        b90_th = sim.G*self.m_BH/v_rms

        ########################### faccio il plot  

        if savefig:

            # Crea sottocartella se non esiste
            Path(scan_folder + sub_folder).mkdir(parents=False, exist_ok=True)

            fig, ax = plt.subplots(figsize=(10,4))
            ax.set_title(f'Velocity dispersion\nslow stars fraction = {self.slow_stars_fraction:.2f}, N = {sim.N}' + subtitle)
            ax.set_xlabel('velocity')
            ax.set_ylabel('number of stars')
            ax.set_xlim(-100,100)
            ax.hist(pv,bins=200,label=f'stars v, std = {pv_std:.2f}',alpha=0.5,color='C0')
            ax.hist(pvx,bins=200,label=f'stars vx, std = {pvx_std:.2f}',alpha=0.2,color='C2') #*100/np.sum(pvx)
            ax.hist(pvy,bins=200,label=f'stars vy, std = {pvy_std:.2f}',alpha=0.2,color='C3')
            ymin, ymax = ax.get_ylim()
            vx_BH = sim.particles['BH'].vx + self.Vx
            ax.vlines(vx_BH,ymin,ymax,'r',linestyles='-',linewidth=4,label=f'vx_BH = {vx_BH:.3f}',alpha=0.5)
            # ax.set_yscale('log')
            ax.set_xscale('symlog')
            ax.legend(loc='upper left')
            plt.savefig(scan_folder + sub_folder + rf'\vd_{frame:d}.png')
            plt.close()

            
            

        return b90_th



    
    def compute_acc_comparison(self,sim,b90_th,b90_fit, time, scan_folder, sub_folder, frame, savefig = False, subtitle = ''):

        rho = self.m_host/((self.boxL*self.nx)*(self.boxL*self.ny)) # surf density
        rho_slow_stars = rho*self.slow_stars_fraction
    
        b_max = self.boxL*self.nx
        Lambda_fit = b_max/b90_fit
        Lambda_th = b_max/b90_th

        ts_run = np.linspace(0,self.T,len(self.v_BH))

        a_b90_fit = -4*np.pi*((sim.G)/self.Vx)**2*self.m_BH*np.log(Lambda_fit)*rho_slow_stars 
        a_th = -4*np.pi*((sim.G)/self.Vx)**2*self.m_BH*np.log(Lambda_th)*rho_slow_stars  

        f = lambda x,m,q: m*x + q
        popt, _ = curve_fit(f,time, self.v_BH)
        a_lin_fit = popt[0]

        def v_th(t,ax_df):
            v_th = self.v_BH[0] + ax_df *t 
            return v_th

        a_sim = np.gradient(self.v_BH, sim.dt)

        # Confronto la velocità del BH della simulazione con la velocità che deriva dalla
        if savefig:

            # Crea sottocartella se non esiste
            Path(scan_folder + sub_folder).mkdir(parents=False, exist_ok=True)

            fig, ax = plt.subplots(figsize=(7,5))
            ax.set_title(f'Confronto accelerazioni' + subtitle)
            ax.plot(ts_run,self.v_BH,'k-',label='sim')
            ax.plot(ts_run,v_th(ts_run,a_th),'C0--',label=f'a_th = {a_th:.2f}')
            ax.plot(ts_run,v_th(ts_run,a_b90_fit),'C1--',label=f'a_b90_fit = {a_b90_fit:.2f}')
            ax.plot(ts_run,v_th(ts_run,a_lin_fit),'C2--',label=f'a_lin_fit = {a_lin_fit:.2f}')
            ax.set_ylabel('velocità')
            ax.set_xlabel('tempo ')
            ax.legend(loc='lower left')
            plt.savefig(scan_folder + sub_folder +rf'\ac_{frame:d}.png')
            plt.close()
            # xmin, xmax = ax[0].get_xlim()

            # ax[1].plot(ts_run,a_sim,'k-',label='sim')
            # ax[1].hlines(a_th,xmin, xmax,colors='C0',linestyles='--',label='th')
            # ax[1].hlines(a_b90_fit,xmin, xmax,colors='C1',linestyles='--',label='b90 fit')
            # ax[1].hlines(a_lin_fit,xmin, xmax,colors='C2',linestyles='--',label='lin fit')
            # ax[0].set_ylabel('velocità')
        
            # ax[1].set_ylabel('accelerazione')
            # ax[1].set_xlabel('tempo ')

            # ax[1].legend()

        return a_th, a_b90_fit, a_lin_fit

def build_setup(boxL,nx, ny, N, nrows, m_host, m_BH, Vx, r, tau, T):
    '''
        Descrivo la costruzione dei self e definisco i vari parametri

    '''



    ncols = int(N/nrows)
   
    xs = np.linspace(-nx*boxL/2,nx*boxL/2,ncols)
    ys = np.linspace(-ny*boxL/2*0.9,ny*boxL/2*0.9,nrows)
    m = 0
    mask = (ys < -m) | (ys > m)
    ys = ys[mask]

    x_BH = []
    v_BH = []
    # x0_BH = boxL*(nx-1)/nx  
    x0_BH = 0
    v0_BH = 0
    x_BH = np.append(x_BH, x0_BH)
    v_BH = np.append(v_BH, v0_BH) 

    return boxL, nx, ny, N, nrows, ncols, m_host, m_BH, x_BH, x0_BH, v_BH, Vx, xs, ys, r, tau    
        

