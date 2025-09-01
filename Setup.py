# importo pacchetti

import pickle
import subprocess
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
 
# Definisco una classe con i parametri della simulazione e le funzioni 
# che utilizzo in heartbeat per lo studio real time della frizione dinamica

class setup_class:
    def __init__(self,boxL:float,nx:int,ny:int,N:int,nrows:int,m_host:float,m_BH:float,Vx:float,T:float,position:str,scale:float|None):
        """
        Classe che funge da utile contenitore per tutte le quantità necessarie alla definizione della classe sim di Rebound.
        Quando setup_class viene chiamata, alcuni suoi attributi (ncols, x_BH, x_BH0, v_BH, xs, ys) vengono costruiti tramite 
        la funzione ausiliaria build_setup. 
        
        Attributes
        ----------
        boxL : float
            Dimensione della box  
            
        nx : int
            Larghezza della box in unità di boxL

        ny : int
            Altezza della box in unità di boxL

        N : int
            Numero di stelle 
            
        nrows : int
            Numero di righe lungo cui sono disposte le stelle
                
        m_host : float
            Massa totale del mare di stelle
        
        m_BH : float
            Massa del buco nero
        
        Vx : float
            Velocità del buco nero
        
        scale : float
            Coefficiente opzionale da specificare quando si utilizza una distribuzione di velocità delle particelle che non è delta-like.
            Viene usato per fare in modo che la dispersione relativa delle distribuzioni di velocità sia la stessa ad ogni Vx.

        T : float
            Tempo finale della simulazione
            
        ncols : int
            Numero di colonne lungo cui sono disposte le particelle iniziali

        x_BH : float
            Coordinata x del buco nero 

        x_BH0 : float 
            Coordinata x del buco nero al tempo iniziale 

        v_BH : float 
            Velocità del buco nero

        xs : ndarray
            Ascisse delle particelle al tempo 0
            
        ys : ndarray     
            Ordinate delle particelle al tempo 0
        """

        boxL, nx, ny, N, nrows, ncols, m_host, m_BH, x_BH, x0_BH, v_BH, Vx, xs, ys, scale, T = build_setup(boxL, nx, ny, N, nrows, m_host, m_BH, Vx, scale, T, position)

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
        self.scale = scale  
        self.T = T

        self.slow_stars_fraction = 1


    def compute_deflection_angle(self, sim, particle_row: list, fig_folder: str, frame: int, savefig=False) -> float:

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

        fig_folder: str
            Cartella in cui vengono salvate le immagini

        frame : int
            Indice crescente che parte da 1 per distinguere in modo semplice le figure associate ad un valore di Vx

        savefig : bool
            Booleano per salvare o meno le figure
            
        Returns
        -------
        b90_fit : float
            Il valore del parametro di impatto critico ottenuto sperimentalmente
        """


        def_angle_extremes = np.zeros(self.nrows)
        def_angles_plot = []
        ys_plot = []
        
        for row in range(0,self.nrows):
            def_angles = []
            for p in sim.particles:
                if p.hash.value != 1753590236  and particle_row[p.hash.value]==row:          # non considero il buco nero, che ha hash 1753590236 
                    
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
            Path(fig_folder).mkdir(parents=False, exist_ok=True)

            fig, ax = plt.subplots()
            ax.plot(ys_plot,def_angles_plot,'C0.',alpha=0.1)
            ax.plot(self.ys,def_angle_extremes,'rx',label='extremes')
            ax.plot(x_fit,y_fit,'k-',alpha=0.5,label=f'b90_fit={b90_fit:.2f}')
            ax.set_title(f'Deflection angle vs impact parameter, Vs = {self.Vx:.2f}')
            ax.set_xlabel('Impact parameter')
            ax.set_ylabel('Deflection angle [deg]')
            ax.legend(loc='upper left')
            plt.savefig(fig_folder + rf'\da_{frame:d}.png')
            plt.close()

        return  b90_fit 




    def compute_velocity_dispersion(self, sim, fig_folder: str, frame: int, savefig=False) -> float:
        """
        Nell'eq di Chandrasekhar compare la frazione di stelle con velocità in modulo minore a quella del BH. Questa funzione la calcola.


        Nel farlo tiene conto del fatto che il numero di stelle nella box non è esattamente costante.
        Infine la funzione può plottare la distribuzione dei moduli delle velocità e delle componenti x e y, con una linea verticale a rappresentare 
        la velocità orizzontale del BH.


        Parameters
        ----------
        sim : class
            Classe costruita tramite rebound
            
        fig_folder: str
            Cartella in cui vengono salvate le immagini

        frame : int
            Indice crescente che parte da 1 per distinguere in modo semplice le figure associate ad un valore di Vx

        savefig : bool
            Booleano per salvare o meno le figure
 
        Returns
        -------
        b90_th : float
            Il valore del parametro di impatto critico calcolato tramite formula teorica
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

        # v_rms = np.mean(vs)     # stavo prendendo sqrt
        # b90_th = sim.G*self.m_BH/v_rms                    #capire che cosa era sbagliato qui
        b90_th = sim.G*self.m_BH/self.Vx**2
        

        ########################### faccio il plot  

        if savefig:

            # Crea sottocartella se non esiste
            Path(fig_folder).mkdir(parents=False, exist_ok=True)

            fig, ax = plt.subplots(figsize=(10,4))
            ax.set_title(f'Velocity dispersion\nslow stars fraction = {self.slow_stars_fraction:.2f}, N = {sim.N}')
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
            plt.savefig(fig_folder + rf'\vd_{frame:d}.png')
            plt.close()

            
            

        return b90_th



    def compute_acc_comparison(self,sim,b90_th: float,b90_fit: float, time: float,fig_folder: str, frame: int, savefig = False) -> tuple[float,float,float]:
        """
     
        
        Questa funzione è il cuore del confronto tra 


        Nel farlo tiene conto del fatto che il numero di stelle nella box non è esattamente costante.
        Infine la funzione può plottare la distribuzione dei moduli delle velocità e delle componenti x e y, con una linea verticale a rappresentare 
        la velocità orizzontale del BH.


        Parameters
        ----------
        sim : class
            Classe costruita tramite rebound
            
        fig_folder: str
            Cartella in cui vengono salvate le immagini

        frame : int
            Indice crescente che parte da 1 per distinguere in modo semplice le figure associate ad un valore di Vx

        savefig : bool
            Booleano per salvare o meno le figure

        Returns
        -------
        a_b90_th : float
            Accelerazione del buco nero calcolata tramite formula di Chandrasekhar e usando come b90 il valore teorico
            
        a_b90_fit : float
            Accelerazione del buco nerocalcolata tramite formula di Chandrasekhar e usando come b90 il valore sperimentale
            
        a_sim : float 
            Accelerazione del buco nero ricavata dal fit lineare della sua velocità

        """

        rho = self.m_host/((self.boxL*self.nx)*(self.boxL*self.ny)) # surf density
        rho_slow_stars = rho*self.slow_stars_fraction
    
        b_max = self.boxL*self.nx
        Lambda_fit = b_max/b90_fit
        Lambda_th = b_max/b90_th

        ts_run = np.linspace(0,self.T,len(self.v_BH))

        a_b90_fit = -4*np.pi*((sim.G)/self.Vx)**2*self.m_BH*np.log(Lambda_fit)*rho_slow_stars 
        a_b90_th = -4*np.pi*((sim.G)/self.Vx)**2*self.m_BH*np.log(Lambda_th)*rho_slow_stars  

        f = lambda x,m,q: m*x + q
        popt, _ = curve_fit(f,time, self.v_BH)
        a_sim = popt[0]

        def v_th(t,ax_df):
            v_th = self.v_BH[0] + ax_df *t 
            return v_th

        # Confronto la velocità del BH della simulazione con la velocità che deriva dalla
        if savefig:

            # Crea sottocartella se non esiste
            Path(fig_folder).mkdir(parents=False, exist_ok=True)

            fig, ax = plt.subplots(figsize=(7,5))
            ax.set_title(f'Confronto accelerazioni, Vs = {self.Vx:.2f}')
            ax.plot(ts_run,self.v_BH,'k-',label='sim')
            ax.plot(ts_run,v_th(ts_run,a_sim),'k--',label=f'a_sim = {a_sim:.2f}')
            ax.plot(ts_run,v_th(ts_run,a_b90_fit),'C1--',label=f'a_b90_fit = {a_b90_fit:.2f}')
            ax.plot(ts_run,v_th(ts_run,a_b90_th),'C2--',label=f'a_b90_th = {a_b90_th:.2f}')

            ax.set_ylabel('velocità')
            ax.set_xlabel('tempo ')
            ax.set_ylim(-2.5,0)
            ax.legend(loc='lower left')
            plt.savefig(fig_folder + rf'\ac_{frame:d}.png')
            plt.close()

        return a_b90_th, a_b90_fit, a_sim


#############################################################################################################################################
#############################################################################################################################################
#############################################################################################################################################

def build_setup(boxL, nx, ny, N, nrows, m_host, m_BH, Vx, scale, T, position):
    """
    Funzione ausiliaria utile a creare e richiamare le quantità necessarie alla definizione della classe sim di Rebound.
    Costruisce gli attributi ncols, x_BH, x_BH0, v_BH, xs, ys della classe setup_class. 

    Parameters
    ----------
    Gli stessi della classe setup_class

    Returns
    -------
    Vedere help setup_class

    """

    ncols = int(N/nrows)
   
    xs = np.linspace(-nx*boxL/2,nx*boxL/2,ncols)            # ascisse delle particelle al tempo 0
    ys = np.linspace(-ny*boxL/2*0.9,ny*boxL/2*0.9,nrows)    # ordinate delle particelle al tempo 0

    x_BH = []
    v_BH = []

    match position:
        case 'right':
            x0_BH = boxL*(nx-1)/nx   
        case 'center':
            x0_BH = 0  
        case _:
            raise ValueError("Parametro 'position' non valido. Usa 'right' oppure 'center'.")

    v0_BH = 0
    x_BH = np.append(x_BH, x0_BH)
    v_BH = np.append(v_BH, v0_BH) 

    return boxL, nx, ny, N, nrows, ncols, m_host, m_BH, x_BH, x0_BH, v_BH, Vx, xs, ys, scale, T



def particles_initial_velocity(dist, Vx, scale = None, plot = False, numb=1):

    """
    Funzione con cui scegliere la distribuzione di velocità iniziale delle particelle. Si possono scegliere 3 tipi di distribuzioni: 'delta', 'gaussian', 'uniform'.
    Nel caso di distribuzioni gaussiane si è deciso di dotarle di dispersione proporzionale al modulo della velocità attorno a cui sono centrate, a meno di un fattore.

    Questo per indagare l'effetto di distribuzione più o meno localizzate in modo indipendente dalla velocitò Vx.    
    Nel progetto è stata utilizzata la distribuzione 'delta'.

    Parameters
    ----------
    dist : str
        Tipo di distribuzione ('delta', 'gaussian', 'uniform')
        
    Vx : float
        Velocità attorno a cui sono centrate le distribuzioni

    scale : float
        Costante di proporzionalità tra sigma e Vx: sigma = |Vx/scale|

    Returns
    -------
    sampled_velocity : float
        Valore di velocità estratto dalla distribuzione scelta 
    """

    match dist:
        case 'delta':
            sampled_velocity = -Vx

        case 'gaussian':
            sigma = np.abs(Vx/scale)
            sampled_velocity = np.random.normal(-Vx, sigma, numb)


    return sampled_velocity




def make_video(frame_dir: str|Path, frame_name: str, output_name: str, framerate: int):
    """
    Questa funzione genera dei filmati .mp4 a partire dalle immagini .png salvate nelle sottocartelle delle varie scansioni. 

    Parameters
    ----------
    frame_dir : str
        Cartella in cui sono salvate le immagini con cui fare il filmato

    frame_name : str
        Template dei nomi delle immagini con cui fale il filmato. Ad esempio 'ac_%d.png'.
    output_name : str
        Nome del filmato .mp4. Ad esempio rf'{scan_folder}\ac_overall.mp4'
    framerate : int

    """

    frame_dir = Path(frame_dir)
    cmd = [
        "ffmpeg",
        "-framerate", str(framerate),
        "-i", str(frame_dir / frame_name),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        str(frame_dir / output_name)
    ]
    subprocess.run(cmd, check=True)





def read_pickle_and_info(pickle_path: str, info_path: str):
    """
    Questa funzione legge i file .pickle e .txt prodotti da... 

    Parameters
    ----------
    pickle_path : str
        Percorso del file .pickle

    info_path : str
        Percorso del file _info.txt
    Returns
    -------
    pickle_file : list
        File contenente le quantità calcolate e salvate durante e a fine simulazioni. 
        b90_th, b90_fit, a_b90_th, a_b90_fit, a_sim, slow_stars_fraction

    info_file : dict
        File contenente un log della simulazione facilmente consultabile e dal quale è possibile in modo automatico ricostruire 
        le griglie m_BH x m_host x Vs per fare i plot dei risultati.

    """

    pickle_file = {}
    info_file = {}

    with open(pickle_path, 'rb') as handle:
        pickle_file = pickle.load(handle)

    with open(info_path, 'r') as file:
        for row in file:
            row = row.strip()
            if row == "" or row.startswith("#"):
                continue  # Salta righe vuote e commenti
            
            if '=' in row:
                key, value = row.split('=')
                key = key.strip()
                value = value.strip()

                # Gestione tuple del tipo (x, y, z)
                if value.startswith('(') and value.endswith(')'):
                    try:
                        elements = value[1:-1].split(',')
                        values_tuple = []
                        for e in elements:
                            e = e.strip()
                            if '.' in e:
                                values_tuple.append(float(e))
                            else:
                                values_tuple.append(int(e))
                        value = tuple(values_tuple)
                    except Exception:
                        pass  # in caso di errore, lo lascia come stringa

                else:
                    # Tenta conversione automatica a int o float se possibile
                    try:
                        if '.' in value:
                            value = float(value)
                        else:
                            value = int(value)
                    except ValueError:
                        pass  # rimane stringa
                
                info_file[key] = value
    
    return pickle_file, info_file







        

